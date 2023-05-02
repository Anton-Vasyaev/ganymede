import dependencies
# python
import time
from typing import Optional
# 3rd party
import onnxruntime as ort
# project
import ganymede.opencv as g_cv
import ganymede.filesystem as g_fs
import ganymede.file       as g_file
from ganymede.ml.onnxruntime import YoloOnnxRuntimeDetector

from ganymede.imaging import NumpyImageHandler, ImageType


def draft_yolo_onnxruntime():
    #cfg_path     = r'D:\data\models\work\unifint\human\darknet\yolov7_tiny_person.cfg'
    #model_path  = r'D:\data\models\work\unifint\human\darknet\yolov7_tiny_person.onnx'

    #cfg_path     = r'D:\data\models\work\unifint\human\darknet\yolov4_tiny_person.cfg'
    #model_path = r'D:\data\models\work\unifint\human\darknet\yolov4_tiny_person.onnx'

    cfg_path   = None
    model_path = r'D:\data\models\work\unifint\human\pytorch\yolov7.onnx'
    
    images_path  = r'D:\data\test_images'


    config : Optional[str] = None
    if not cfg_path is None:
        with open(cfg_path, 'r') as fh:
            config = fh.read()

    #providers = ['CPUExecutionProvider']
    providers = ['TensorrtExecutionProvider']

    detector = YoloOnnxRuntimeDetector(model_path, config, providers=providers)

    images_path_list = g_fs.rsearch_files(str(images_path))

    img = g_cv.imread(images_path_list[0])

    img_handler = NumpyImageHandler(img, ImageType.BGR)

    batch_detections = detector.detect(
        [img_handler] * 4,
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8]
    )

    for detections in batch_detections:
        show_img = g_cv.resize_frame(img, (1600, 900))
        for detection in detections:
            g_cv.draw_bbox(show_img, detection.bbox, (255, 0, 0), 2)
        g_cv.imshow('debug', show_img)

    times_n    = 25
    batch_size = 4

    detections = detector.detect([img_handler] * batch_size, [0.4] * batch_size, [0.25] * batch_size)
    
    start = time.time()
    for idx in range(times_n):
        print(f'calculate performance:{idx+1}/{times_n}')
        detections = detector.detect(
            [img_handler] * batch_size, 
            [0.25] * batch_size, 
            [0.4]  * batch_size
        )
    end = time.time()

    print(f'process time per image:{(end - start) / (times_n * batch_size)} sec')
    

    for img_path in images_path_list:
        img = g_cv.imread(str(img_path))

        img_handler = NumpyImageHandler(img, ImageType.BGR)

        detections = detector.detect([img_handler], [0.25], [0.25])[0]

        show_img = img.copy()
        show_img = g_cv.resize_frame(show_img, (1600, 900))

        for detection in detections:
            g_cv.draw_bbox(show_img, detection.bbox, (255, 0, 0), 2)
            
            g_cv.draw_text(show_img, f'{detection.class_id}', detection.bbox[0:2], (255, 0, 0))

        g_cv.imshow('debug', show_img)


if __name__ == '__main__':
    draft_yolo_onnxruntime()