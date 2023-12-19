# python
import time
from typing import Optional
from datetime import datetime
# 3rd party
import numpy as np
from ganymede.video import CvVideoReader
# project
from .data.read_frame_data import ReadFrameData


class CvRealTimeVideoReader:
    __far_read_data : ReadFrameData

    __start_read_time : int

    __start_read_timestamp : int

    __reader : CvVideoReader

    __speed : float


    def __get_current_time(self) -> int:
        return int(time.time() * 1_000_000)


    def __init__(self, path : str, speed : float = 1.0):
        self.__far_read_data = ReadFrameData(np.array([]), -1)

        self.__start_read_time = 0
        self.__start_read_timestamp = 0

        self.__reader = CvVideoReader(path)

        self.__speed = speed


    def read(self) -> Optional[ReadFrameData]:
        frame, timestamp = self.__reader.read()

        if frame is None:
            return None
        
        return ReadFrameData(frame, timestamp)
    

    def set_position(self, seconds : float):
        set_timestamp = seconds * 1_000_000

        while True:
            read_frame_data = self.read()
            if read_frame_data is None:
                break

            if read_frame_data.timestamp > set_timestamp:
                self.__start_read_time = 0
                self.__start_read_timestamp = read_frame_data.timestamp

                self.__far_read_data = read_frame_data
                break

            del read_frame_data.frame


    def speed_read(self) -> Optional[ReadFrameData]:
        if self.__start_read_time == 0:
            if self.__far_read_data is None:
                frame, timestamp = self.__reader.read()

                if frame is None:
                    return None
                
                self.__far_read_data = ReadFrameData(frame, timestamp)
                self.__start_read_timestamp = timestamp

            self.__start_read_time = self.__get_current_time()

        prev_data = self.__far_read_data
        while True:
            current_time = self.__get_current_time()
            prev_timestamp_duration = prev_data.timestamp - self.__start_read_timestamp
            time_duration = current_time - self.__start_read_time

            if time_duration > prev_timestamp_duration:
                frame, timestamp = self.__reader.read()
                if frame is None:
                    break
                
                current_read_data = ReadFrameData(frame, timestamp)

                current_time = self.__get_current_time()
                current_timestamp_duration = current_read_data.timestamp - self.__start_read_timestamp
                time_duration = current_time - self.__start_read_time

                if current_timestamp_duration > time_duration:
                    self.__far_read_data = current_read_data

                    return prev_data
                
                prev_data = current_read_data