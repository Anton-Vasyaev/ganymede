from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import numpy as np
import time
import zmq
import torch

import cv2 as cv

if __name__ == '__main__':
    # Настройки
    CONFIDENCE = 0.5
    NMS_THRESHOLD = 0.4
    IMG_SIZE = 416

    # Путь до конфига
    config_path = "loco\\yolov3-tiny2.cfg"
    # Путь до весов
    weights_path = "loco\\yolov3-tiny_best.weights"
    # Путь до лейблов
    labels_path = "loco\\labels_ioco.txt"

    # Установка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Launch models on device: {device}")

    # Загрузка модели
    model = Darknet(config_path, img_size=IMG_SIZE).to(device)
    model.load_darknet_weights(weights_path)

    model.eval()

    # Биндинг сокета
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")
    print("Server started @ tcp://*.5558")

    while True:
        print('Waiting for request...')
        image_data = socket.recv()
        image_meta = socket.recv_json()
        print(image_meta)

        start = time.perf_counter()

        # Загрузка изображения
        #img = np.frombuffer(image_data, dtype=np.uint8).reshape(image_info['height'],
                                                                #image_info['width'],
                                                                #image_info['channels'])

        # Подгон под вход модели
        #img = transforms.ToTensor()(img).to(device)
        #img, _ = pad_to_square(img, 0)
        #img = resize(img, IMG_SIZE).unsqueeze(0)

        image_count = image_meta['images_count']
        images_meta = image_meta['images_meta']

        # take images from frame
        propagate_images = []
        images_data_offset = 0
        for idx in range(image_count):
            image_info = images_meta[idx]

            present_type = image_info['present_type']

            read_img = None
            if present_type == 'compressed':
                im_size = image_info['image_size']
                im_buffer = np.frombuffer(
                    image_data,
                    np.uint8,
                    im_size,
                    images_data_offset
                )
                decode_img = cv.imdecode(im_buffer, cv.IMREAD_UNCHANGED)
                read_img = decode_img
                images_data_offset += im_size

            elif present_type == 'uncompressed':
                im_w = image_info['image_width']
                im_h = image_info['image_height']
                im_stride = image_info['image_stride']

                im_size = im_h * im_w * 3
                # im_size = im_h * im_stride
                im_buffer = np.frombuffer(
                    image_data,
                    np.uint8,
                    im_size,
                    images_data_offset
                    )
                ''' # THIS FOR SITUATION WHEN IMAGES WITH STRIDE
                im_buffer.shape = (im_h, im_stride)
                image = im_buffer[:,0:im_w*3]
                image.shape = (im_h, im_w, 3)
                '''
                im_buffer.shape = (im_h, im_w, 3)
                read_img = im_buffer
                images_data_offset += im_size

        #print(image_data.shape)
        # Оригинальный размер изображения
        #image_size = (image_info['height'], image_info['width'])
		
        read_img = transforms.ToTensor()(read_img).to(device)
        read_img, _ = pad_to_square(read_img, 0)
        read_img = resize(read_img, IMG_SIZE).unsqueeze(0)
		
        response = []

        detect_start = time.perf_counter()
        with torch.no_grad():
			
            detections = model(read_img)
            detections = non_max_suppression(detections, CONFIDENCE, NMS_THRESHOLD)
            print(len(detections[0].shape))
            
        detect_end = time.perf_counter()

        if len(detections[0].shape) != 1:
            # Подгон под размеры реального изображения
            detections = rescale_boxes(detections[0], IMG_SIZE, im_size)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_w = x2 - x1
                box_h = y2 - y1

                """res = {
                    'class_id': int(cls_pred),
                    'conf': cls_conf.item(),
                    'bbox': {
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(box_w),
                        'height': int(box_h)
                    }
                }"""
                res = {
                    "position": {
                        "xtl": int(x1)/int(im_w),
                        "ytl": int(y1)/int(im_h),
                        "w": int(box_w)/int(im_w),
                        "h": int(box_h)/int(im_h)
                        },
                    "class_id": int(cls_pred),
                    "object_confidence": cls_conf.item()
                    }
                response.append(res)
        else:
            print("Detections is None")

        final = {'loco': response}
        stop = time.perf_counter()
        print('Process time: %f ms; detect time: %f ms' % ((stop - start) / 1000000,
                                                           (detect_end - detect_start) / 1000000))

        socket.send_json(final)

        # Отправка json-а в сервис агрегации
        socket_agr = context.socket(zmq.REQ)
        socket_agr.connect("tcp://localhost:5559")
        socket_agr.send_json(final)
        message = socket_agr.recv_json()
        print(message)
