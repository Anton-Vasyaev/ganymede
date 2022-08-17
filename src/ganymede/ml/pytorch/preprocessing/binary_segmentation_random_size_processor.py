# python
import random
from typing import Tuple
# 3rd party
import cv2 as cv
# project
import ganymede.random             as g_random
import ganymede.imaging            as g_image
import ganymede.ml.pytorch.tensor  as g_tensor
from ganymede.imaging                 import ImageType
from ....dataset.processing.auxiliary import default_input_processor



class BinarySegmentationRandomSizeProcessor:
    def __init__(
        self,
        img_type    : ImageType,
        size_range  : Tuple[int, int],
        random_seed : int = 1024, 
        input_processor=default_input_processor
    ):
        self.size_range = size_range

        self.img_type = img_type

        self.random_instance = random.Random(random_seed)

        self.input_processor = input_processor


    def __call__(self, batch_example):
        resize_s = int(g_random.get_random_distance(self.size_range[0], self.size_range[1]))

        img_list, mask_list = batch_example

        new_img_list  = []
        new_mask_list = []

        for img, mask in zip(img_list, mask_list):
            if self.img_type == ImageType.RGB:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            elif self.img_type == ImageType.GRAY:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                pass

            img  = cv.resize(img, (resize_s, resize_s), interpolation=cv.INTER_AREA)
            g_image.create_channel_if_not_exist(img)

            mask = cv.resize(mask, (resize_s, resize_s), interpolation=cv.INTER_NEAREST)

            new_img_list.append(img)
            new_mask_list.append(mask)

        img_t = g_tensor.img_list_to_tensor_batch(new_img_list, normalized=False)

        if not self.input_processor is None: 
            process_result = self.input_processor(img_t)
            if not process_result is None: img_t = process_result

        mask_t = g_tensor.img_list_to_tensor_batch(new_mask_list, normalized=True)
        mask_t[mask_t > 0.01] = 1.0

        return img_t, mask_t