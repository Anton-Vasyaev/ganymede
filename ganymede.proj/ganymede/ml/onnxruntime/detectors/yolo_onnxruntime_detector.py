# python
from typing import List, Any, Optional, cast
# 3rd party
import cv2   as cv
import numpy as np
import onnxruntime as ort

import ganymede.core.type_aux as type_aux
from ganymede.ml.data import *


from nameof import nameof
# project
import ganymede.imaging as g_img
from ganymede.imaging import ImageShape, ImageType, NumpyImageHandler
from ganymede.opencv import get_cv_color_conversion_code

from ganymede.native.auxml.funcs import *

import ganymede.ml.onnxruntime.validation as ort_valid

from ...darknet import *

from ganymede.native.auxml.funcs import process_yolo_sealead_output_detections


@dataclass
class DarknetConfigData:
    config_backbone : DarknetBackbone
    
    outputs_params : List[Any]


    def __init__(self, config_backbone : DarknetBackbone):
        self.config_backbone = config_backbone

        self.outputs_params = list()
        for layer in config_backbone.layers:
            if isinstance(layer, YoloLayer):
                self.outputs_params.append(layer)
            else:
                continue

        if len(self.outputs_params) == 0:
            raise Exception(f'not find output layer in darknet config.')
        
        if not type_aux.is_equal_types_b(self.outputs_params):
            raise Exception(
                f'different output layers types in darknet config.')


class YoloOnnxRuntimeDetector:
#region data

    __session : ort.InferenceSession

    __darknet_config_data : Optional[DarknetConfigData]

    __current_batch_size : int
    
    __input_size : ImageShape

    __input_image_type : ImageType

    __input_name : str

    __aux_mat_size : int

    __input_batch : np.ndarray

    __resize_mat : np.ndarray

    __color_conversion_mat : np.ndarray

#endregion

#region construct_and_destruct

    def __init__(
        self,
        model_path           : str,
        darknet_config       : Optional[str] = None,
        aux_mat_size         : int           = 1024,
        batch_size           : int           = 1,
        input_image_type     : ImageType     = ImageType.UNKNOWN,
        providers            : Optional[Any] = ['CPUExecutionProvider']
    ):
        self.__session = ort.InferenceSession(model_path, providers=providers)

        self.__darknet_config_data = None
        if not darknet_config is None:
            config_backbone = read_darknet_bone_from_str(darknet_config)
            self.__darknet_config_data = DarknetConfigData(config_backbone)
        
        self.__aux_mat_size       = aux_mat_size
        self.__current_batch_size = batch_size
        self.__input_image_type   = input_image_type

        self.__validate_params()
        self.__init_params()
        self.__init_aux_mats()

#endregion

#region private_methods

    def __validate_params(self):
        ort_valid.number_of_inputs_equal(self.__session, 1, nameof(YoloOnnxRuntimeDetector))
        inputs = self.__session.get_inputs()
        ort_valid.is_convolutional_shape_format(inputs[0], nameof(YoloOnnxRuntimeDetector))

        net_c : int = 0

        if not self.__darknet_config_data is None:
            config_backbone = self.__darknet_config_data.config_backbone
            config_backbone.validate()

            net_w = config_backbone.net_params.width
            net_h = config_backbone.net_params.height
            net_c = config_backbone.net_params.channels

            ort_valid.is_equal_image_shape(inputs[0], net_w, net_h, net_c, nameof(YoloOnnxRuntimeDetector))

            outputs = self.__session.get_outputs()
            for output in outputs:
                ort_valid.is_convolutional_shape_format(output, nameof(YoloOnnxRuntimeDetector))

        else:
            ort_valid.number_of_outputs_equal(self.__session, 1, nameof(YoloOnnxRuntimeDetector))
            output = self.__session.get_outputs()[0]
            ort_valid.number_of_dimensions_equal(output, 3, nameof(YoloOnnxRuntimeDetector))

            net_c = inputs[0].shape[1]

        if self.__input_image_type != ImageType.UNKNOWN:
            img_channels = self.__input_image_type.get_channels()
            if img_channels != net_c:
                raise Exception(
                    f'channels of network != channels of image type \'{self.__input_image_type}\':'
                    f'{net_c} != {img_channels}.'
                )


    def __init_params(self):
        if not self.__darknet_config_data is None:
            network_bone = self.__darknet_config_data.config_backbone

            net_w = network_bone.net_params.width
            net_h = network_bone.net_params.height
            net_c = network_bone.net_params.channels

            self.__input_size = ImageShape(net_w, net_h, net_c)

        else:
            inputs = self.__session.get_inputs()
            input = inputs[0]
            b, c, h, w = input.shape
            
            self.__input_size = ImageShape(w, h, c)

            outputs = self.__session.get_outputs()
            output  = outputs[0]
            
            b, bbox_count, val_count = output.shape

        self.__input_name = cast(str, self.__session.get_inputs()[0].name)

        if self.__input_image_type == ImageType.UNKNOWN:
            net_c = self.__input_size.channels
            if net_c == 1:
                self.__input_image_type = ImageType.GRAY
            elif net_c == 3:
                self.__input_image_type = ImageType.BGR
            else:
                raise Exception(f'Not have default {nameof(ImageType)} for channels:{net_c}.')
            

    def __init_aux_mats(self):
        batch_size = self.__current_batch_size

        in_w, in_h, in_c = self.__input_size.decompose()

        self.__input_batch = np.zeros((batch_size, in_h, in_w, in_c), dtype=np.float32)
        self.__resize_mat = np.zeros((in_h, in_w, in_c), dtype=np.uint8)

        aux_size = self.__aux_mat_size
        self.__color_conversion_mat = np.zeros((aux_size, aux_size, 3), dtype=np.uint8)


    def __resize_batch(self, batch_size : int):
        in_w, in_h, in_c = self.__input_size.decompose()

        self.__current_batch_size = batch_size
        self.__input_batch.resize((batch_size, in_h, in_w, in_c), refcheck=False)


    def __postprocessing_darknet_config(
        self, 
        net_output        : List[np.ndarray],
        object_thresholds : List[float],
        nms_thresholds    : List[float]
    ) -> ObjectDetectionBatch:
        output_params = self.__darknet_config_data.outputs_params
        if isinstance(output_params[0], YoloLayer):
            yolo_output_params = cast(List[YoloLayer], output_params)

            preds = cast(List[np.ndarray], net_output)
            preds = [p.transpose(0, 2, 3, 1).copy() for p in preds]

            detections = process_yolo_detections(
                yolo_output_params,
                preds,
                self.__darknet_config_data.config_backbone.net_params,
                object_thresholds,
                nms_thresholds
            )

            return detections
        else:
            raise Exception(f'not implemented postrocessing for output layer:{type(output_params[0])}.')
        
    
    def __postprocessing_sealed_output(
        self,
        net_output        : List[np.ndarray],
        object_thresholds : List[float],
        nms_thresholds    : List[float]
    ) -> ObjectDetectionBatch:
        in_w, in_h, in_c = self.__input_size.decompose()
        
        preds = cast(np.ndarray, net_output[0])

        detections_batch = process_yolo_sealead_output_detections(
            preds,
            object_thresholds,
            nms_thresholds,
            in_w,
            in_h
        )

        return detections_batch


#endregion

#region methods

    def detect(
        self, 
        input_batch       : List[NumpyImageHandler],
        object_thresholds : List[float],
        nms_thresholds    : List[float]
    ) -> ObjectDetectionBatch:
        if len(input_batch) == 0:
            raise Exception('image batch is empty')
        
        self.__resize_batch(len(input_batch))

        in_w, in_h, in_c = self.__input_size.decompose()

        # make batch
        for img_idx in range(len(input_batch)):
            image_handler = input_batch[img_idx]

            image_type    = image_handler.image_type
            image         = image_handler.image

            img_h, img_w = image.shape[0:2]

            selected_image        = image
            color_conversion_code = get_cv_color_conversion_code(image_type, self.__input_image_type)
            if not color_conversion_code is None:
                self.__color_conversion_mat.resize((img_h, img_w, in_c))
                cv.cvtColor(image, cv.COLOR_GRAY2RGB, self.__color_conversion_mat)
                selected_image = self.__color_conversion_mat

            cv.resize(selected_image, (in_w, in_h), self.__resize_mat, interpolation=cv.INTER_AREA)

            np.copyto(self.__input_batch[img_idx], self.__resize_mat)

        self.__input_batch /= 255.0

        input_batch_t = self.__input_batch.transpose(0, 3, 1, 2)

        # propagate
        net_output = self.__session.run(None, { self.__input_name : input_batch_t})

        # postprocessing
        detections : ObjectDetectionBatch = list()
        if not self.__darknet_config_data is None:
            detections = self.__postprocessing_darknet_config(net_output, object_thresholds, nms_thresholds)
        else:
            detections = self.__postprocessing_sealed_output(net_output, object_thresholds, nms_thresholds)

        return detections

#endregion