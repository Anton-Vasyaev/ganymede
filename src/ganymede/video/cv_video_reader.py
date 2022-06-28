# python
import os
# 3rd party
import cv2 as cv


class CvVideoReader:
    def __init__(self, path):
        if not os.path.exists(path):
            raise Exception(f'cannot open video_file, reason: not exist, path:{path}')

        self.capture = cv.VideoCapture(path)

        self.current_position = 0
    

    def get_frame_count(self):
        return int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))


    def get_frame_pos(self):
        return int(self.capture.get(cv.CAP_PROP_POS_FRAMES))


    def get_fps(self):
        return int(self.capture.get(cv.CAP_PROP_FPS))


    def get_position(self): return self.current_position


    def set_position(self, position):
        self.current_position = position

        self.capture.set(cv.CAP_PROP_POS_MSEC, position / 1000)


    def read(self):
        ret, frame = self.capture.read()

        if not ret: return None, None

        position = self.capture.get(cv.CAP_PROP_POS_MSEC)
        self.current_position = position * 1000

        return frame, self.current_position


    def skip_and_read(self, msecs):
        need_position = self.current_position + msecs

        while True:
            frame, position = self.read()
            if frame is None: return None, None

            position = int(position)
            if position >= need_position:
                return frame, position
