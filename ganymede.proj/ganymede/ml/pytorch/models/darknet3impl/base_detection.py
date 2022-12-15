# python
from __future__ import division

import os
import sys
import ast
import time
import datetime
import argparse

from enum import IntEnum

# 3rd party
import cv2 as cv
from pathlib import Path

import torch
import torch.nn.functional as torch_f

from lpr.lpr_openvino_x4 import LprOpenVinoV1
from lpr.mobilenet_v2_lpr import MobileNetV2
from lpr.decode_metrics import decode_output, decode_tensor_output

from models import *

from utils.nn_fusion import *
from utils.video import *
from utils.image import *
from utils.visual_debug import *
from utils.models import *

from melting_detection import SimpleMeltingDetector


class MainType(IntEnum):
    LADLE       = 0
    TRUCK       = 1
    SLAGTRUCK   = 2


def normalize(detections, current_shape, clamp=False):
    curr_h, curr_w = current_shape[0:2]

    norm_list = []
    for detection in detections:
        if detection.shape == torch.Size([0]):
            norm_list.append(detection)
            continue
        detection = detection.clone()

        detection[:,0] /= curr_w
        detection[:,2] /= curr_h
        detection[:,1] /= curr_w
        detection[:,3] /= curr_h

        if clamp:
            detection[:,0:4] = torch.clamp(detection[:,0:4], 0.0, 1.0)

        norm_list.append(detection)

    return norm_list


def offset_detections(detections, relative_offset=0.8):
    offset_list = []

    for detection in detections:
        if detection.shape == torch.Size([0]):
            offset_list.append(detection)
            continue
        detection = detection.clone()
        w_offset = (detection[:,2] - detection[:,0]) * relative_offset
        h_offset = (detection[:,3] - detection[:,1]) * relative_offset

        detection[:, 0] -= w_offset
        detection[:, 2] += w_offset
        detection[:, 1] -= h_offset
        detection[:, 3] += h_offset

        offset_list.append(detection)

    return offset_list


def get_crops(detections, object_type):
    crops = []

    for curr_detections in detections:
        img_crop = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in curr_detections[0]:
            if int(cls_pred) == object_type:
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w, h = x2 - x1, y2 - y1

                ob_conf     = float(conf)
                cls_conf    = float(cls_conf)
                img_crop.append([x1, y1, w, h, cls_conf, ob_conf])
        crops.append(img_crop)

    return crops


def get_number_crops(
    number_detections,
    ladle_crops
):
    new_detections = []

    for img_idx in range(len(number_detections)):
        image_crop  = ladle_crops[img_idx]
        number_image_result = number_detections[img_idx]

        new_img_result = []

        for ladle_idx in range(len(number_image_result)):
            ladle_crop = image_crop[ladle_idx]
            l_x1, l_y1, l_w, l_h, _, _ = ladle_crop
            number_ladle_result = number_image_result[ladle_idx]

            new_ladle_result = []

            for number_idx in range(len(number_ladle_result)):
                number = number_ladle_result[number_idx]
                x1, y1, x2, y2, _, _, _ = number
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                x1, y1 = l_x1 + x1 * l_w, l_y1 + y1 * l_h
                x2, y2 = l_x1 + x2 * l_w, l_y1 + y2 * l_h
                w = x2 - x1
                h = y2 - y1
                new_ladle_result.append([x1, y1, w, h])

            new_img_result.append(new_ladle_result)
        new_detections.append(new_img_result)

    return new_detections


class YoloDetector:
    @staticmethod
    def has_empty_crops(
        crops
    ):
        empty_status = True
        for crop in crops:
            if len(crop) != 0:
                empty_status = False
                break

        return empty_status


    def __init__(
        self,
        darknet_model,
        img_size,
        nms_thres,
        conf_thres,
        relative_offset = 0.0,
        batch_size = 8,
    ):
        self.darknet_model  = darknet_model
        self.img_size       = img_size
        self.nms_thres      = nms_thres
        self.conf_thres     = conf_thres

        self.relative_offset    = relative_offset
        self.batch_size         = batch_size

        device = next(self.darknet_model.parameters()).device

        self.batch_cache = torch.zeros(
            (self.batch_size, self.darknet_model.input_channels, self.img_size, self.img_size),
            dtype=torch.float32,
            device=device
        )

    def __call__(
        self,
        images,
        images_crops = None
    ):
        device = next(self.darknet_model.parameters()).device

        # initialize accumulation list
        detections = []
        for idx in range(len(images)):
            detections.append([])


        # make crops list is (image_idx, tlx, tly, w, h), coord is normalize (0.0, 1.0)
        crops_list = []
        if images_crops is None:
            for img_idx in range(len(images)):
                crops_list.append([float(img_idx), 0.0, 0.0, 1.0, 1.0])
        else:
            img_idx = 0
            for current_crops in images_crops:
                for crop in current_crops:
                    crops_list.append([float(img_idx)] + crop)
                img_idx += 1


        if YoloDetector.has_empty_crops(crops_list):
            return detections
        # propagate
        cache_img_idx = int(crops_list[0][0])

        load_img_cache = select_or_make_tensor(images[cache_img_idx], device)

        offset_idx = 0
        while offset_idx < len(crops_list):
            curr_batch_size = min(self.batch_size, len(crops_list) - offset_idx)

            # make batch from crop list
            curr_crops = crops_list[offset_idx:offset_idx+curr_batch_size]
            sample_idx = 0

            for crop in curr_crops:
                img_idx = int(crop[0])
                if img_idx != cache_img_idx:
                    load_img_cache =  select_or_make_tensor(images[img_idx], device)
                    cache_img_idx = img_idx

                # cut image and interpolate
                img_h, img_w = load_img_cache.shape[2:4]
                crop_x, crop_y, crop_w, crop_h = crop[1:5]
                crop_x, crop_y = int(crop_x * img_w), int(crop_y * img_h)
                crop_w, crop_h = int(crop_w * img_w), int(crop_h * img_h)

                img = torch_f.interpolate(
                    load_img_cache[:, :, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w],
                    (self.img_size, self.img_size),
                    mode='area'
                )
                if self.darknet_model.input_channels == 1:
                    img = cast_tensor_rgb_to_gray(img)
                self.batch_cache[sample_idx] = img
                sample_idx += 1

            curr_detections = self.darknet_model(self.batch_cache[0:curr_batch_size])

            curr_detections = non_max_suppression(
                curr_detections, self.conf_thres, self.nms_thres
            )
            if self.relative_offset != 0.0:
                curr_detections = offset_detections(curr_detections, self.relative_offset)

            curr_detections = normalize(
                curr_detections, (self.img_size, self.img_size), clamp=True
            )

            for idx in range(curr_batch_size):
                img_idx = int(curr_crops[idx][0])
                detections[img_idx].append(curr_detections[idx])

            offset_idx += self.batch_size
        return detections


        '''
        original_image = images[0]
        show_image = cv.resize(images[0], (self.img_size, self.img_size), cv.INTER_AREA)
        image = show_image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image.shape = (1,) + image.shape
        image = torch.from_numpy(image).to(device)

        detections = self.darknet_model(image)
        detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        detections = torch.stack(detections)
        detections = normalize(detections, (self.img_size, self.img_size), clamp=True)

        debug_yolo_detections(
            original_image,
            detections[0]
        )
        return detections
        '''


class LprDetector:
    def __init__(self, lpr_model, input_size, batch_size=8):
        self.lpr_model  = lpr_model
        self.batch_size = batch_size
        self.input_size = input_size + (self.lpr_model.img_channels,)


    def __call__(self, images, number_crops):
        device = next(self.lpr_model.parameters()).device

        # initalize accumulation list
        lpr_detections  = []
        lpr_confidences = []
        for img_idx in range(len(number_crops)):
            img_crop = number_crops[img_idx]
            image1_detections = []
            image2_detections = []

            for ladle_idx in range(len(img_crop)):
                ladle_crop = img_crop[ladle_idx]
                ladle_detections = []

                for number_idx in range(len(ladle_crop)):
                    ladle_detections.append(-1)
                image1_detections.append(ladle_detections.copy())
                image2_detections.append(ladle_detections.copy())
            lpr_detections.append(image1_detections)
            lpr_confidences.append(image2_detections)

        # initialize crop list
        # contain [ img_idx, ladle_idx, number_idx, x1, y1, w, h]
        crops_list = []
        for img_idx in range(len(number_crops)):
            image_result = number_crops[img_idx]

            for ladle_idx in range(len(image_result)):
                ladle_result = image_result[ladle_idx]

                for number_idx in range(len(ladle_result)):
                    number = ladle_result[number_idx]
                    x1, y1, w, h = number
                    crops_list.append(
                        [img_idx, ladle_idx, number_idx, x1, y1, w, h]
                    )

        input_h, input_w, input_c = self.input_size

        batch_cache = torch.zeros(
            (self.batch_size, input_c, input_h, input_w),
            dtype=torch.float32,
            device=device
        )

        if YoloDetector.has_empty_crops(crops_list):
            return lpr_detections, lpr_confidences

        cache_img_idx = crops_list[0][0]
        load_img_cache = select_or_make_tensor(images[cache_img_idx], device)

        offset_idx = 0
        while offset_idx < len(crops_list):
            curr_batch_size = min(self.batch_size, len(crops_list) - offset_idx)

            # make batch from crop list
            curr_crops = crops_list[offset_idx:offset_idx+curr_batch_size]
            sample_idx = 0
            for crop in curr_crops:
                img_idx = int(crop[0])
                if img_idx != cache_img_idx:
                    load_img_cache =  select_or_make_tensor(
                        images[img_idx], device
                    )
                    cache_img_idx = img_idx

                # cut image and interpolate
                img_h, img_w = load_img_cache.shape[2:4]
                crop_x, crop_y, crop_w, crop_h = crop[3:8]
                crop_x, crop_y = int(crop_x * img_w), int(crop_y * img_h)
                crop_w, crop_h = int(crop_w * img_w), int(crop_h * img_h)

                img = torch_f.interpolate(
                    load_img_cache[:, :, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w],
                    (input_h, input_w),
                    mode='area'
                )
                if self.lpr_model.img_channels == 1:
                    img = cast_tensor_rgb_to_gray(img)
                batch_cache[sample_idx] = img
                sample_idx += 1

            detections = self.lpr_model(batch_cache[0:curr_batch_size])


            decodes, confidences = decode_tensor_output(detections, 10)

            for decode_idx in range(len(decodes)):
                select_crop = curr_crops[decode_idx]
                img_idx, ladle_idx, number_idx, _, _, _, _ = select_crop

                select_decode = decodes[decode_idx]
                if len(select_decode) == 0:
                    convert_int = -1
                else:
                    convert_int = int("".join([str(i) for i in select_decode]))

                lpr_detections[img_idx][ladle_idx][number_idx] = convert_int
                lpr_confidences[img_idx][ladle_idx][number_idx] = confidences[decode_idx]

            offset_idx += self.batch_size

        return lpr_detections, lpr_confidences


class GlobalDetector:
    def __init__(
        self,
        main_detector,
        number_detector,
        lpr_detector = None
    ):
        self.main_detector     = main_detector
        self.number_detector   = number_detector
        self.lpr_detector      = lpr_detector


    def __call__(
        self,
        images
    ):
        device = next(self.main_detector.darknet_model.parameters()).device
        tensor_images = []

        for image in images:
            tensor_images.append(select_or_make_tensor(image, device))

        main_detections = self.main_detector(tensor_images)

        ladle_crops = get_crops(main_detections, MainType.LADLE)
        truck_crops = get_crops(main_detections, MainType.TRUCK)


        number_detections = self.number_detector(tensor_images, ladle_crops)

        number_crops = get_number_crops(
            number_detections,
            ladle_crops
        )

        lpr_numbers, confidences = self.lpr_detector(tensor_images, number_crops)

        result_images = []
        for image_idx in range(len(images)):
            ladles = []
            ladle_crop = ladle_crops[image_idx]

            image_numbers = number_detections[image_idx]
            for ladle_idx in range(len(ladle_crop)):
                l_x1, l_y1, l_w, l_h, cls_conf, ob_conf = ladle_crop[ladle_idx]
                ladle = {
                    'position' : {
                        'xtl'   : l_x1,
                        'ytl'   : l_y1,
                        'w'     : l_w,
                        'h'     : l_h
                    },
                    'class_confidence'  : cls_conf,
                    'object_confidence' : ob_conf
                }
                ladle_numbers = image_numbers[ladle_idx]

                numbers = []
                for number_idx in range(len(ladle_numbers)):
                    lpr_number          = lpr_numbers[image_idx][ladle_idx][number_idx]
                    number_confidence   = confidences[image_idx][ladle_idx][number_idx]
                    ladle_number    = ladle_numbers[number_idx]
                    x1, y1, x2, y2, conf, cls_conf, cls_pred = ladle_number
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    conf, cls_conf, cls_pred = float(conf), float(cls_conf), float(cls_pred)
                    w = x2 - x1
                    h = y2 - y1

                    x1, y1 = l_x1 + x1 * l_w, l_y1 + y1 * l_h
                    x2, y2 = l_x1 + x2 * l_w, l_y1 + y2 * l_h
                    w = x2 - x1
                    h = y2 - y1

                    number = {
                        'position' : {
                            'xtl'   : x1,
                            'ytl'   : y1,
                            'w'     : w,
                            'h'     : h,
                        },
                        'id'    : lpr_number,
                        'number_confidence' : list(map(lambda x: float(x), number_confidence)),
                        'object_confidence' : conf
                    }
                    numbers.append(number)

                ladle['numbers'] = numbers
                ladles.append(ladle)

            truck_crop = truck_crops[image_idx]
            trucks = []
            for truck in truck_crop:
                x1, y1, w, h, cls_conf, ob_conf = truck
                trucks.append(
                    {
                        'position' : {
                            'xtl'   : x1,
                            'ytl'   : y1,
                            'w'     : w,
                            'h'     : h
                        },
                        'class_confidence' : cls_conf,
                        'object_confidence' : ob_conf
                    }
                )

            result_image = {
                'ladles' : ladles,
                'trucks' : trucks
            }
            result_images.append(result_image)

        return { 'images' : result_images }




def get_service_detectors_dict(config):
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir_path = config.models_dir_path
    # Set up model
    main_model = Darknet(
        os.path.join(models_dir_path, 'side', 'yolov3_nlmk.cfg'),
        img_size=416
    ).to(device)
    main_model.load_darknet_weights(
        os.path.join(models_dir_path, 'side', 'yolov3_nlmk.weights')
    )
    main_model.eval()
    main_model = fuse_bn_recursively(main_model)

    truck_model = Darknet(
        os.path.join(models_dir_path, 'overview', 'yolov3_overview.cfg'),
        img_size=416
    ).to(device)
    truck_model.load_darknet_weights(
        os.path.join(models_dir_path, 'overview', 'yolov3_overview.weights')
    )
    truck_model.eval()
    truck_model = fuse_bn_recursively(truck_model)

    number_model = Darknet(
        os.path.join(models_dir_path, 'numbers', 'yolov3_tiny_numbers.cfg'),
        img_size=416
    ).to(device)
    number_model.load_darknet_weights(
        os.path.join(models_dir_path, 'numbers', 'yolov3_tiny_numbers.weights')
    )
    number_model.eval()
    number_model = fuse_bn_recursively(number_model)

    lpr_model = MobileNetV2(
        1,
        11
    ).to(device)
    lpr_model.load_state_dict(torch.load(
        os.path.join(models_dir_path, 'lpr', 'mobilenet_v2.torch')
    ))
    lpr_model.eval()
    lpr_model = fuse_bn_recursively(lpr_model)

    main_detector = YoloDetector(
        main_model,
        416,
        config.nms_thres,
        config.conf_thres,
        0.0,
        2
    )

    number_detector = YoloDetector(
        number_model,
        416,
        config.nms_thres,
        config.conf_thres,
        0.04,
        4
    )

    truck_detector = YoloDetector(
        truck_model,
        416,
        config.nms_thres,
        config.conf_thres
    )

    lpr_detector = LprDetector(
        lpr_model,
        (128, 128),
        4
    )

    global_detector = GlobalDetector(
        main_detector,
        number_detector,
        lpr_detector
    )

    detectors_dict =  {
        'global_detector'   : global_detector,
        'truck_detector'    : truck_detector
    }

    preheat_models(detectors_dict)

    return detectors_dict


def process_detection(detectors_dict, images):
    global_detector     = detectors_dict['global_detector']
    truck_detector      = detectors_dict['truck_detector']

    side_imgs = []
    overview_imgs = []

    melting_idxs = []

    img_idx = 0
    for img in images:
        if img[1] == 'side':
            side_imgs.append((img[0], img_idx))
        elif img[1] == 'overview':
            overview_imgs.append((img[0], img_idx))

        if len(img) >= 3:
            melting_info = img[2]
            if not melting_info is None:
                melting_idxs.append((img_idx, melting_info))

        img_idx += 1

    side_result = global_detector([img[0] for img in side_imgs])

    truck_detections = truck_detector([img[0] for img in overview_imgs])

    # melting detection
    melting_detections = []
    for melting_idx, (melting_area, min_val, min_area) in melting_idxs:
        melting_status = SimpleMeltingDetector.static_detect_melting(
            images[melting_idx][0],
            melting_area,
            min_val,
            min_area
        )
        melting_detections.append((melting_idx, melting_status))

    final_result = {
        'images' : []
    }
    # initialize array
    for idx in range(len(images)):
        final_result['images'].append(
            {
                'ladles' : [],
                'trucks' : [],
                'melting': None
            })

    # append result for side cameras
    for side_idx in range(len(side_imgs)):
        img_idx = side_imgs[side_idx][1]
        final_result['images'][img_idx] = side_result['images'][side_idx]

    # append result for overview cameras
    for overview_idx in range(len(overview_imgs)):
        img_idx = overview_imgs[overview_idx][1]

        trucks = []
        ladles = []
        current_detections = truck_detections[overview_idx]
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in current_detections[0]:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            w, h = x2 - x1, y2 - y1

            ob_conf     = float(conf)
            cls_conf    = float(cls_conf)
            
            detect_object = {
                'position' : {
                    'xtl'   : x1,
                    'ytl'   : y1,
                    'w'     : w,
                    'h'     : h
                },
                'class_confidence'  : cls_conf,
                'object_confidence' : ob_conf
            }

            if int(cls_pred) == 0:
                detect_object['numbers'] = []
                ladles.append(detect_object)
            else:
                trucks.append(detect_object)

        final_result['images'][img_idx]['ladles'] = ladles
        final_result['images'][img_idx]['trucks'] = trucks

    for image in final_result['images']:
        image['melting'] = None
    # append result for melting detections
    for melting_idx, melting_status in melting_detections:
        final_result['images'][melting_idx]['melting'] = melting_status

    return final_result


def start_processing(
    options
):
    detectors_dict = get_service_detectors_dict(options)

    '''
    images = []
    for idx in range(6):
        new_idx = idx + 1
        images.append(cv.imread(f'data/nlmk_samples/{new_idx}.png'))

    result = global_detector(images)

    debug_global_result(
        images,
        result
    )
    '''
    #cap1 = cv.VideoCapture(r"Z:\2020\НЛМК - Тестовые ролики Сталевозы\К2\Фрагмент_15_1\20200618_005715_Cam_172.23.11.124.ts")

    images = []
    start_timestamp = (1 * 60 + 20) * 1000
 
    print(f'drop frames')
    #drop_frames(cap1, minutes=2, seconds=0)
    #drop_frames(cap2, minutes=3, seconds=20)

    images_dir = Path(r'C:\data\nlmk\dumps\2020_jpg')

    for file_path in images_dir.iterdir():
        parent_dir = file_path.parent
        splits = str(file_path.stem).split('_')

        if splits[2] != 'Left':
            continue

        left_img_path       = parent_dir / f'{splits[0]}_{splits[1]}_Left_{splits[3]}.jpg'
        right_img_path      = parent_dir / f'{splits[0]}_{splits[1]}_Right_{splits[3]}.jpg'
        overview_img_path   = parent_dir / f'{splits[0]}_{splits[1]}_Overview_{splits[3]}.jpg'

        left_frame      = cv.imread(str(left_img_path))
        right_frame     = cv.imread(str(right_img_path))
        overview_frame  = cv.imread(str(overview_img_path))
        
        images = [
            [
                left_frame,
                'side',
                None
            ],
            [
                right_frame,
                'side',
                None
            ],
            [
                overview_frame,
                'overview',
                None
            ]
        ]

        start = time.time()
        result = process_detection(detectors_dict, images)
        end = time.time()
        print(f'processing:{end - start} seconds')

        debug_global_result(
            [img[0] for img in images],
            result,
            True
        )

        print(f'result:{result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir_path', type=str, default='config/nlmk')
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument('--nms_thres', type=float, default=0.4)
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")

    opt = parser.parse_args()

    start_processing(opt)