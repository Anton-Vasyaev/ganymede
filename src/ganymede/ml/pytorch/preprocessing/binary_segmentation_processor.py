# 3rd party
import cv2 as cv
# project
import ganymede.imaging            as g_image
import ganymede.ml.pytorch.tensor  as g_tensor
from ganymede.imaging                 import ImageType
from ....dataset.processing.auxiliary import default_input_processor



class BinarySegmentationProcessor:
    def __init__(
        self,
        input_size,
        img_type,
        input_processor=default_input_processor
    ):
        self.input_size      = input_size
        self.img_type        = img_type
        self.input_processor = input_processor


    def __call__(self, batch_example):
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

            img  = cv.resize(img, self.input_size, interpolation=cv.INTER_AREA)
            g_image.create_channel_if_not_exist(img)

            mask = cv.resize(mask, self.input_size, interpolation=cv.INTER_AREA)

            new_img_list.append(img)
            new_mask_list.append(mask)

        img_t = g_tensor.img_list_to_tensor_batch(new_img_list, normalized=False)

        if not self.input_processor is None: 
            process_result = self.input_processor(img_t)
            if not process_result is None: img_t = process_result

        mask_t = g_tensor.img_list_to_tensor_batch(new_mask_list, normalized=True)
        mask_t[mask_t > 0.01] = 1.0

        return img_t, mask_t