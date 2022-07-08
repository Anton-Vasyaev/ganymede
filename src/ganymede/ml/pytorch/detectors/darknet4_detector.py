# python
from typing      import List
# 3rd party
import torch
import cv2 as cv
# 3rd party
import ganymede.imaging as g_img
# project
from ganymede.ml.pytorch.models.darknet_impl.darknet2pytorch    import Darknet
from ganymede.ml.pytorch.models.darknet_impl.torch_utils        import do_detect
from ganymede.ml.data                                           import ObjectDetection





class Darknet4Detector:
    def __init__(
        self, 
        config_path  : str, 
        weights_path : str, 
        use_gpu      : bool = True
    ):
        self.model = Darknet(config_path)
        self.model.load_weights(weights_path)
        self.model.eval()

        if use_gpu and torch.cuda.is_available():
            device = torch.cuda.device('cuda:0')
            self.model.to(device)

        self.input_channels = int(self.model.blocks[0]['channels'])


    def detect(
        self, 
        img, 
        confidence_threshold = 0.5,
        nms_threshold        = 0.4
    ) -> List[ObjectDetection]:
        with torch.no_grad():
            if g_img.get_channels(img) == 1 and self.input_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif g_img.get_channels(img) == 3 and self.input_channels == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif g_img.get_channels(img) == 3 and self.input_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            img = cv.resize(img, (self.model.width, self.model.height))
            g_img.create_channel_if_not_exist(img)

            detections = []

            boxes = do_detect(self.model, img, confidence_threshold, nms_threshold, self.use_cuda)

            for detection in boxes[0]:
                x1, y1, x2, y2, object_confidence, _, class_id = detection

                x1, y1 = float(x1), float(y1)
                x2, y2 = float(x2), float(y2)

                object_confidence = float(object_confidence)
                class_id          = int(class_id)

                detection.append(
                    ObjectDetection(
                        [x1, y1, x2, y2],
                        class_id,
                        object_confidence
                    )
                )

            return detections
