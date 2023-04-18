import dependencies
# python
import ctypes
import time
from typing import List, cast
# 3rd party
import numpy as np
import cv2 as cv
import torch
from pathlib import Path

import ganymede.opencv as g_cv
import ganymede.imaging as g_img
import ganymede.draw    as g_draw
import ganymede.ml.pytorch.tensor as g_tensor
import ganymede.filesystem as g_fs
# test
from ganymede.ml.pytorch.detectors.darknet_torch_detector import DarknetTorchDetector



def call_visualise_detections():
    #cfg_path     = r'/home/anton/datasets/yolo_test/model/yolov4_tiny.cfg'
    #weights_path = r'/home/anton/datasets/yolo_test/model/yolov4_tiny.weights'
    #images_path = r'/home/anton/datasets/yolo_test/test_images'

    cfg_path     = r'D:\data\models\work\unifint\human\darknet\yolov7_tiny_person.cfg'
    weights_path = r'D:\data\models\work\unifint\human\darknet\yolov7_tiny_person.weights'

    #cfg_path     = r'D:\data\models\work\unifint\human\darknet\yolov4_tiny_person.cfg'
    #weights_path = r'D:\data\models\work\unifint\human\darknet\yolov4_tiny_person_final.weights'
    
    images_path  = r'D:\data\test_images'

    detector = DarknetTorchDetector.load_from_files(cfg_path, weights_path, False)

    images_path_list = g_fs.rsearch_files(str(images_path))

    img = g_cv.imread(images_path_list[0])

    batch_detections = detector.detect(
        [img, img, img, img],
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8]
    )

    for detections in batch_detections:
        show_img = g_cv.resize_frame(img, (1600, 900))
        for detection in detections:
            g_cv.draw_bbox(show_img, detection.bbox, (255, 0, 0), 2)
        g_cv.imshow('debug', show_img)

    detections = detector.detect([img], [0.4], [0.25])
    times_n    = 25
    batch_size = 4
    start = time.time()
    for idx in range(times_n):
        print(f'calculate performance:{idx+1}/{times_n}')
        detections = detector.detect(
            [img]  * batch_size, 
            [0.25] * batch_size, 
            [0.4]  * batch_size
        )
    end = time.time()

    print(f'process time per image:{(end - start) / (times_n * batch_size)} sec')
    

    for img_path in images_path_list:
        img = g_cv.imread(str(img_path))

        detections = detector.detect([img], [0.25], [0.25])[0]

        show_img = img.copy()
        show_img = g_cv.resize_frame(show_img, (1600, 900))

        for detection in detections:
            g_cv.draw_bbox(show_img, detection.bbox, (255, 0, 0), 2)
            
            g_cv.draw_text(show_img, f'{detection.class_id}', detection.bbox[0:2], (255, 0, 0))

        g_cv.imshow('debug', show_img)


if __name__ == '__main__':
    call_visualise_detections()