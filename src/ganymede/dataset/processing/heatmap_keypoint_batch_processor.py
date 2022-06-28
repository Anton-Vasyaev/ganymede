# 3rd party
import cv2 as cv
import torch
# project
import ganymede.imaging.image  as g_image
import ganymede.ml.pytorch.tensor as g_tensor
from   ganymede.imaging.image import ImageType
from .heatmap    import generate_keypoints_heatmap
from .auxiliary import default_input_processor


class HeatmapKeypointBatchProcessor:
    def __init__(
        self, 
        sigma, 
        input_size,
        img_type,
        heatmap_size    = None,
        input_processor = default_input_processor
    ):
        self.sigma       = sigma
        
        self.input_size = input_size
        self.img_type   = img_type

        self.heatmap_size = heatmap_size
        if self.heatmap_size is None: self.heatmap_size = input_size


    def __call__(
        self,
        batch
    ):
        img_list, keypoints_list = batch

        img_batch_list      = []
        heatmap_tensor_list = []

        for img, keypoints in zip(img_list, keypoints_list):    
            if self.img_type == ImageType.RGB:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            elif self.img_type == ImageType.GRAY:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            img     = cv.resize(img, self.input_size, interpolation=cv.INTER_AREA)
            heatmap = generate_keypoints_heatmap(self.heatmap_size, keypoints, self.sigma)

            img_batch_list.append(img)
            heatmap_tensor_list.append(heatmap)

        img_batch     = g_tensor.img_list_to_tensor_batch(img_batch_list)
        heatmap_batch = torch.concat(heatmap_tensor_list)

        return img_batch, heatmap_batch

        

            
