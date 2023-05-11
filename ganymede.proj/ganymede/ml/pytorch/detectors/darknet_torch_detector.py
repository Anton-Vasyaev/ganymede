# python
from typing import List, cast
# 3rd party
import torch
import cv2   as cv
import numpy as np
import ganymede.ml.pytorch.tensor as g_tensor
import ganymede.imaging as g_img
from ganymede.ml.data import *
# project
from ...darknet                       import DarknetModel
from ..models.darknet.model.modules         import YoloModule, DarknetLayerModule
from ganymede.native.auxml.funcs.yolo_layer import process_yolo_detections
from ..models.darknet.postprocessing.yolo_layer import postprocessing_yolo_layer


class DarknetTorchDetector:
    model : DarknetModel

    device : torch.device

    output_modules : List[DarknetLayerModule]

    @staticmethod
    def load_from_files(
        cfg_path     : str, 
        weights_path : str,
        use_gpu      : bool = False
    ):
        with open(cfg_path, 'r') as fh:
            cfg_data = fh.read()

            with open(weights_path, 'rb') as fh:
                weights_data = fh.read()

                return DarknetTorchDetector(cfg_data, weights_data, use_gpu)

    @staticmethod
    def load_from_data(
        cfg_data     : str, 
        weights_data : bytes,
        use_gpu      : bool = False
    ):
        return DarknetTorchDetector(cfg_data, weights_data, use_gpu)


    def __init__(
        self,
        config_data  : str,
        weights_data : bytes,
        use_gpu      : bool = True
    ):
        self.model = DarknetModel.load_from_str(config_data)
        self.model.load_weights_from_binary(weights_data)
        self.model.train(False)

        self.output_modules = self.model.get_output_modules()

        self.device = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)


    def __validate_input_params(
        self,
        images         : List[np.ndarray],
        obj_thresholds : List[float],
        nms_thresholds : List[float]
    ):
        if len(images) != len(obj_thresholds) or len(images) != len(nms_thresholds):
            raise Exception(
                f'lengths of images, object threshold and nms threshold are not same:'
                f'{len(images)}, {len(obj_thresholds)}, {len(nms_thresholds)}.'
            )


    def detect(
        self, 
        images         : List[np.ndarray], 
        obj_thresholds : List[float], 
        nms_thresholds : List[float]
    ) -> ObjectDetectionBatch:
        self.__validate_input_params(
            images,
            obj_thresholds,
            nms_thresholds
        )

        net_w = self.model.net_params.width
        net_h = self.model.net_params.height
        net_c = self.model.net_params.channels

        images_batch_list : List[np.ndarray] = list()

        for img in images:
            img_c = g_img.get_channels(img)

            if img_c == 1 and net_c == 3:
                img = g_img.cast_one_channel_img(img, 3)
            elif img_c == 3 and net_c == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif img_c == 4 and net_c == 1:
                img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
            elif img_c == 4 and net_c == 3:
                img = img[...,0:3]

            img = cv.resize(img, (net_w, net_h), interpolation=cv.INTER_AREA)

            if len(img.shape) == 2:
                img = img.reshape(img.shape + (1,))

            images_batch_list.append(img)

        img_batch = g_tensor.img_list_to_tensor_batch(images_batch_list)
        img_batch = img_batch.to(self.device)

        self.model(img_batch)

        output_modules = self.output_modules
        if isinstance(output_modules[0], YoloModule):
            output_yolo_modules = cast(List[YoloModule], output_modules)

            output_layers_params = [out.params for out in output_yolo_modules]

            outputs = [out.output.detach().cpu().permute(0, 2, 3, 1).numpy() for out in output_yolo_modules]

            detection_batch = process_yolo_detections(
                output_layers_params,
                outputs,
                self.model.net_params,
                obj_thresholds,
                nms_thresholds
            )

            return detection_batch

        else:
            raise Exception(f'Not implement postprocessing for output type:{type(output_modules[0])}')