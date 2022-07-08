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


class RandomSizeClassSegmentationProcessor:
    def __init__(
        self,
        class_n         : int,
        height_range    : Tuple[float, float],
        width_range     : Tuple[float, float],
        size_scale      : float = 0,
        ranom_seed      : int   = 1024,
        convert_to_brob : bool  = False
    ):
        self.class_n = class_n

        self.random_instance = random.Random(ranom_seed)
        
        self.height_range = height_range
        self.width_range  = width_range
        self.size_scale   = size_scale

        self.convert_to_prob = convert_to_brob


    def __call__(
        self,
        batch
    ):
        img_list, class_map_list = batch

        input_h = g_random.get_random_distance(
            self.height_range[0], 
            self.height_range[1], 
            self.random_instance
        )
        if self.size_scale != 0:
            input_w = input_h * self.size_scale
        else:
            input_w = g_random.get_random_distance(
                self.width_range[0],
                self.width_range[1],
                self.random_instance
            )
        input_w, input_h = int(input_w), int(input_h)

        new_img_list       = []
        new_class_map_list = []
        for img, class_map in zip(img_list, class_map_list):
            img       = cv.resize(img,       (input_w, input_h), interpolation=cv.INTER_AREA)
            class_map = cv.resize(class_map, (input_w, input_h), interpolation=cv.INTER_NEAREST)

            new_img_list.append(img)
            new_class_map_list.append(class_map)

        img_batch    = g_tensor.img_list_to_tensor_batch(new_img_list)
        target_batch = g_tensor.img_list_to_tensor_batch(new_class_map_list, normalized=False)
        target_batch = target_batch.type(torch.int32)

        if self.convert_to_prob:
            target_batch = class_map_to_prob_map(target_batch)


        return img_batch, target_batch