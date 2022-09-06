# python
from typing import Tuple, Any
# 3rd party
import torch
import cv2 as cv
import numpy as np
# project
import ganymede.imaging as g_img
import ganymede.ml.pytorch.tensor as g_tensor


class OneClassSegmentationDetector:
    def __init__(
        self, 
        model       : torch.nn.Module, 
        input_size  : Tuple[int, int] = None, 
        in_channels : int             = None,
        threshold   : float           = 0.5,
        activation  : Any             = torch.sigmoid
    ):
        self.model      = model
        self.model.train(False)
        self.input_size = input_size
        if in_channels is None:
            self.in_channels = self.model.in_channels
        else:
            self.in_channels = in_channels

        self.threshold = threshold

        self.activation = activation


    def detect(self, img : np.ndarray) -> np.ndarray:
        img_h, img_w = img.shape[0:2]

        in_w, in_h = self.input_size

        with torch.no_grad():
            if not self.input_size is None:
                in_w, in_h = self.input_size
                img = cv.resize(img, (in_w, in_h), interpolation=cv.INTER_AREA)

            if g_img.get_channels(img) == 1 and self.in_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif g_img.get_channels(img) == 3 and self.in_channels == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif g_img.get_channels(img) == 3 and self.in_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            img_batch = g_tensor.img_list_to_tensor_batch([img])

            output = self.model(img_batch)
            output = self.activation(output)

            output = g_tensor.tensor_batch_to_img_list(output)[0]

            output = (output > self.threshold).astype(np.uint8) * 255

            output = cv.resize(output, (img_w, img_h), interpolation=cv.INTER_NEAREST)

            return output