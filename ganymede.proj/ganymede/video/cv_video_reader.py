# python
import os
from typing import Tuple, Optional
# 3rd party
import numpy as np
import cv2 as cv # type: ignore


ReadResult = Tuple[Optional[np.ndarray], Optional[int]]

class CvVideoReader:
    __capture : cv.VideoCapture


    def __init__(self, path : str):
        if not os.path.exists(path):
            raise Exception(f'cannot open video_file, reason: not exist, path:{path}')

        self.capture = cv.VideoCapture(path)

        self.current_position = 0
    

    def get_capture(self) -> cv.VideoCapture:
        return self.__capture
    

    def get_frame_count(self) -> int:
        return int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))


    def get_frame_pos(self) -> int:
        return int(self.capture.get(cv.CAP_PROP_POS_FRAMES))


    def get_fps(self) -> int:
        return int(self.capture.get(cv.CAP_PROP_FPS))


    def get_position(self) -> int: return self.current_position


    def set_position(self, position : int):
        self.current_position = position

        self.capture.set(cv.CAP_PROP_POS_MSEC, position / 1000)


    def read(self) -> ReadResult:
        ret, frame = self.capture.read()

        if not ret: return None, None

        position = self.capture.get(cv.CAP_PROP_POS_MSEC)
        self.current_position = int(position * 1000)

        return frame, self.current_position


    def skip_and_read(self, msecs) -> ReadResult:
        need_position = self.current_position + msecs

        while True:
            frame, position = self.read()
            if frame is None: return None, None

            position = int(position)
            if position >= need_position:
                return frame, position
