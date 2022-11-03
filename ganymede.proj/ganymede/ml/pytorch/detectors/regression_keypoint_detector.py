# python
from typing import List
# 3rd party
import torch
import numpy as np
import cv2   as cv
# project
import ganymede.ml.pytorch.tensor as g_tensor
import ganymede.imaging           as g_img

from ganymede.math.point2 import Point2D


class RegressionKeypointDetector:
    def __init__(
        self, 
        model, 
        input_size,
        add_coords = False
    ):
        self.model = model
        self.model.train(False)
        self.device = next(self.model.parameters())

        add_channels = 0
        if add_coords: add_channels = 2
        self.in_channels = model.in_channels - add_channels

        self.input_size = input_size

        self.add_coords = add_coords


    def detect(
        self, 
        img : np.ndarray
    ) -> List[Point2D]:
        if len(img.shape) >= 3:
            if img.shape[2] == 3 and self.in_channels == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = cv.resize(img, self.input_size, interpolation=cv.INTER_AREA)

        if   g_img.get_channels(img) == 1 and self.in_channels == 3:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif g_img.get_channels(img) == 3 and self.in_channels == 1:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_h, img_w = img.shape[0:2]

        input_batch = g_tensor.img_list_to_tensor_batch([img])
        if self.add_coords:
            coords_batch = g_tensor.make_coords_input_tensor(1, img_h, img_w)
            input_batch  = torch.cat([input_batch, coords_batch], 1)

        input_batch = input_batch.to(self.device)

        output = self.model(input_batch)
        output = torch.sigmoid(output)

        points = output.cpu().detach().numpy().tolist()[0]

        return points


    def __call__(self, img):
        return self.detect(img)