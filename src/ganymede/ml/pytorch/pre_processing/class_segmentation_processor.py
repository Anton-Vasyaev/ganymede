# python
import random
from typing import Tuple
# 3rd party
import torch
import cv2 as cv

import ganymede.ml.pytorch.tensor as g_tensor
import ganymede.opencv            as g_cv
import ganymede.random            as g_random

# project
from .auxiliary import class_map_to_prob_map


class ClassSegmentationProcessor:
    def __init__(
        self,
        input_size      : Tuple[int, int],
        class_n         : int,
        convert_to_brob : bool  = False
    ):
        self.input_size      = input_size
        self.class_n         = class_n
        self.convert_to_prob = convert_to_brob


    def __call__(
        self,
        batch
    ):
        img_list, class_map_list = batch

        new_img_list       = []
        new_class_map_list = []
        for img, class_map in zip(img_list, class_map_list):
            img       = cv.resize(img,       self.input_size, interpolation=cv.INTER_AREA)
            class_map = cv.resize(class_map, self.input_size, interpolation=cv.INTER_NEAREST)

            new_img_list.append(img)
            new_class_map_list.append(class_map)

        img_batch    = g_tensor.img_list_to_tensor_batch(new_img_list)
        target_batch = g_tensor.img_list_to_tensor_batch(new_class_map_list, normalized=False)
        target_batch = target_batch.type(torch.int32)

        if self.convert_to_prob:
            target_batch = class_map_to_prob_map(target_batch)
        else:
            b, _, h, w = target_batch.shape
            target_batch = target_batch.view(b, h, w).type(torch.long)


        return img_batch, target_batch