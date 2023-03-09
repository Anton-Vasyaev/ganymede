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
from ganymede.ml.pytorch.detectors.darknet_detector import DarknetDetector



def call_visualise_detections():
    cfg_path     = r'/home/anton/datasets/yolo_test/model/yolov4-tiny.cfg'
    weights_path = r'/home/anton/datasets/yolo_test/model/yolov4-tiny.weights'
    images_path = r'/home/anton/datasets/yolo_test/test_images'

    #cfg_path     = r'C:\data\models\shkr\horizontal_roll\2022.10.31\yolov4_tiny_shkr.cfg'
    #weights_path = r'C:\data\models\shkr\horizontal_roll\2022.10.31\yolov4_tiny_shkr.weights'
    #images_path  = r'C:\datasets\shkr\test_horizontal_darknet_impl'

    detector = DarknetDetector.load_from_files(cfg_path, weights_path, True)

    images_path_list = g_fs.rsearch_files(str(images_path))

    img = g_cv.imread(images_path_list[0])

    #detections = detector.detect(img, 0.4, 0.25)
    #times_n    = 100

    '''
    start = time.time()
    for _ in range(times_n):
        detections = detector.detect(img, 0.25, 0.4)
    end = time.time()

    print(f'process time per forward:{(end - start) / times_n} sec')
    '''

    for img_path in images_path_list:
        img = g_cv.imread(str(img_path))

        detections = detector.detect(img, 0.4, 0.25)

        show_img = img.copy()
        show_img = g_cv.resize_frame(show_img, (1600, 900))

        for detection in detections:
            g_cv.draw_bbox(show_img, detection.bbox, (255, 0, 0), 2)
            g_cv.draw_text(show_img, f'{detection.class_id}', detection.bbox[0:2], (255, 0, 0))

        g_cv.imshow('debug', show_img)


if __name__ == '__main__':
    call_visualise_detections()