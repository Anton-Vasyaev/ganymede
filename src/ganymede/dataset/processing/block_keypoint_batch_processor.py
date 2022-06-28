# importing
import os
import sys
sys.path.append(os.path.abspath('./../../ganymede.python'))
# 3rd party
import torch
import numpy as np
import cv2 as cv
# project
import ganymede.imaging.image as g_image
import ganymede.ml.pytorch.tensor as g_tensor
from ganymede.imaging.image import ImageType


def clip(x, min_v, max_v):
    return max(min_v, min(x, max_v))


class BlockKeypointBatchProcessor:
    @staticmethod
    def merge_coords(coords_list):
        arr = np.array(coords_list, dtype=np.float32)

        return torch.from_numpy(arr)


    def __init__(
        self,
        img_size,
        img_type,
        add_coord_channels = True,
        additional_detector = None
    ):
        self.img_size = img_size
        self.img_type = img_type
        
        self.add_coord_channels = add_coord_channels

        self.detector = additional_detector


    def cut_from_detector(self, img, points):
        img_h, img_w = img.shape[0:2]

        l, t, r, b = self.detector(img)

        det_w      = r - l
        det_h      = b - t

        new_points = []
        for point in points:
            x, y = point
            x, y = (x - l) / det_w, (y - t) / det_h

            new_points.append((x, y))

        x1, y1 = int(l * img_w), int(t * img_h)
        x2, y2 = int(r * img_w), int(b * img_h)

        x1, y1 = clip(x1, 0, img_w), clip(y1, 0, img_h)
        x2, y2 = clip(x2, 0, img_w), clip(y2, 0, img_h)

        new_img = img[y1:y2, x1:x2]

        return new_img, new_points


    def __call__(
        self,
        batch_example
    ):
        img_list, coords_list = batch_example

        resize_img_list = []
        new_coords_list = []
        for img, points in zip(img_list, coords_list):
            if self.img_type == ImageType.RGB:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            elif self.img_type == ImageType.GRAY:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                pass

            if not self.detector is None:
                img, points = self.cut_from_detector(img, points)

            img  = cv.resize(img, self.img_size, interpolation=cv.INTER_AREA)
            g_image.create_channel_if_not_exist(img)

            resize_img_list.append(img)
            new_coords_list.append(points)

        img_batch  = g_tensor.img_list_to_tensor_batch(resize_img_list)
        b, _, h, w = img_batch.shape

        if self.add_coord_channels:
            coords_batch = g_tensor.make_coords_input_tensor(b, h, w)
            input_batch  = torch.cat([img_batch, coords_batch], 1)
        else:
            input_batch = img_batch

        coords_batch = BlockKeypointBatchProcessor.merge_coords(new_coords_list)

        return input_batch, coords_batch
