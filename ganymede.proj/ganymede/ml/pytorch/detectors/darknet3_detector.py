# python
from typing      import List
# 3rd party
import torch
import cv2 as cv
# project
import ganymede.ml.pytorch.tensor as g_tensor
import ganymede.imaging           as g_img

from ganymede.ml.pytorch.models.darknet3impl.models         import Darknet
from ganymede.ml.pytorch.models.darknet3impl.utils.utils    import non_max_suppression
from ganymede.ml.data                                       import ObjectDetection


class Darknet3Detector:
    def __init__(
        self,
        cfg_path     : str,
        weights_path : str,
        use_gpu      : bool = False
    ):
        self.model = Darknet(cfg_path)
        self.model.load_darknet_weights(weights_path)
        self.model.eval()

        if use_gpu and torch.cuda.is_available():
            device = torch.cuda.device('cuda:0')
            self.model.to(device)

        self.input_channels = self.model.input_channels
        self.img_size       = self.model.img_size


    def detect(
        self, 
        img,
        confidence_threshold = 0.5,
        nms_threshold        = 0.4
    ) -> List[ObjectDetection]:
        with torch.no_grad():
            img = cv.resize(img, (self.img_size, self.img_size), interpolation=cv.INTER_AREA)

            if g_img.get_channels(img) == 1 and self.input_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif g_img.get_channels(img) == 3 and self.input_channels == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif g_img.get_channels(img) == 3 and self.input_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            img_batch = g_tensor.img_list_to_tensor_batch([img])

            detections = self.model(img_batch)
            detections = non_max_suppression(detections, confidence_threshold, nms_threshold)
            detections = detections[0].cpu().detach().numpy().tolist()


            convert_detections = []
            for x1, y1, x2, y2, object_conf, class_conf, class_id in detections:
                x1, y1 = x1 / self.img_size, y1 / self.img_size
                x2, y2 = x2 / self.img_size, y2 / self.img_size
                cls_id      = int(class_id)
                object_conf = float(object_conf)
                class_conf  = float(class_conf)

                convert_detections.append(
                    ObjectDetection(
                        [x1, y1, x2, y2],
                        cls_id,
                        object_conf,
                        class_conf
                    )
                )

            return convert_detections