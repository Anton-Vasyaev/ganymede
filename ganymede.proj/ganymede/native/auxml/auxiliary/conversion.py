# python
import ctypes
from ctypes import *
# 3rd party
from ganymede.ml.data import *
# project
from ganymede.native.auxml.interop.data import object_type, object_handler
from ganymede.native.auxml.interop.api  import *
from .validating import validate_return_status


def get_detections_from_handler(detections_batch_handler : object_handler) -> ObjectDetectionBatch:
    batch_size_c = c_uint64()
    return_status = api_detections_batch_size(
        detections_batch_handler,
        pointer(batch_size_c)
    )
    validate_return_status(return_status)
    batch_size = batch_size_c.value

    detections_batch : ObjectDetectionBatch = []

    for batch_idx in range(batch_size):
        batch_idx_c = c_uint64(batch_idx)

        detections_count_c = c_uint64()
        return_status = api_detections_batch_detections_count(
            detections_batch_handler,
            batch_idx_c,
            byref(detections_count_c)
        )
        validate_return_status(return_status)

        detections_count = detections_count_c.value
        object_detections_arr = (object_detection * detections_count)()
        return_status = api_detections_batch_detections_store(
            detections_batch_handler,
            batch_idx_c,
            object_detections_arr
        )

        detections : ObjectDetectionList = []

        for obj_det in object_detections_arr:
            x1 : float = obj_det.x1
            y1 : float = obj_det.y1
            x2 : float = obj_det.x2
            y2 : float = obj_det.y2

            class_id : int = obj_det.class_id

            obj_conf : float = obj_det.object_confidence
            cls_conf : float = obj_det.class_confidence

            detections.append(ObjectDetection((x1, y1, x2, y2), class_id, obj_conf, cls_conf))

        detections_batch.append(detections)

    return detections_batch
