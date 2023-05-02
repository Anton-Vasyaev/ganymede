# python
from typing import List
# 3rd party
import numpy as np
import cv2   as cv
# project
from ganymede.ml.data import ObjectDetectionList, ObjectDetection

def make_yolo_v5_predictions(
    preds                : np.ndarray,
    input_width          : int,
    input_height         : int,
    nms_threshold        : float,
    confidence_threshold : float,
    score_threshold      : float
) -> ObjectDetectionList:
    class_indices = []
    confidences = []
    boxes = []

    rows_count = preds.shape[0]

    for r in range(rows_count):
        row = preds[r]

        confidence = row[4]

        if confidence >= confidence_threshold:
            classes_scores = row[5:]
            _, _, _, max_indx = cv.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > score_threshold):

                confidences.append(confidence)

                class_indices.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()

                x /= input_width
                y /= input_height

                w /= input_width
                h /= input_height

                x1 = x - 0.5 * w
                y1 = y - 0.5 * h
                box = np.array([x1, y1, w, h])
                boxes.append(box)

    indexes : List[int] = cv.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

    object_detections : ObjectDetectionList = list()

    for idx in indexes:
        obj_conf = confidences[idx]
        class_idx = class_indices[idx]

        # skip no human objects
        if class_idx != 0:
            continue

        box = boxes[idx]

        x1, y1, w, h = box

        x2, y2 = x1 + w, y1 + h

        #x1, y1 = x1 * w_scale, y1 * h_scale
        #x2, y2 = x2 * w_scale, y2 * h_scale

        object_detections.append(ObjectDetection([x1, y1, x2, y2], class_idx, obj_conf, 0.0))

    return object_detections