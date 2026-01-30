# python
import os
from typing import Tuple, Optional
# 3rd party
import numpy as np
import cv2 as cv # type: ignore
from .data.read_frame_data import ReadFrameData


class CvVideoReader:
    __capture : cv.VideoCapture


    def __init__(self, path : str):
        if os.path.isdir(path):
            raise Exception(f'cannot open video file, reason: is dir, path:{path}')
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


    def read(self) -> Optional[ReadFrameData]:
        
        ret, frame = self.capture.read()

        if not ret: return None

        position = self.capture.get(cv.CAP_PROP_POS_MSEC)
        self.current_position = int(position * 1000)

        return ReadFrameData(frame, self.current_position)


    def skip_and_read(self, msecs) -> Optional[ReadFrameData]:
        need_position = self.current_position + msecs

        while True:
            read_data = self.read()
            if read_data is None: return None
            position  = read_data.timestamp

            position = int(position)
            if position >= need_position:
                return read_data
