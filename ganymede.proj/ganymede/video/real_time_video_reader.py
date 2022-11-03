# python
import time
# project
from .cv_video_reader import CvVideoReader


class RealTimeVideoReader:
    @staticmethod
    def get_time(): return int(time.time() * 1000 * 1000)


    def __init__(
        self, 
        path  : str, 
        speed : float = 1.0
    ):
        self.reader = CvVideoReader(path)

        self.speed = speed

        self.last_position  = None
        self.last_read_time = None

        self.launch_flag = False
        self.play_flag   = False
        self.stop_time   = None


    def start(self):
        if self.launch_flag: raise Exception('video has started')
        
        self.last_read_time = int(RealTimeVideoReader.get_time() * 1000)
        self.last_position = 0
        self.launch_flag = True
        self.play_flag   = True


    def stop(self):
        if not self.play_flag: raise Exception('video is not playing')

        self.play_flag = False
        self.stop_time = self.get_time()


    def resume(self):
        if self.play_flag: raise Exception('video is playing')

        stop_duration = self.stop_time - self.last_read_time

        self.last_read_time = self.get_time() - stop_duration
        self.play_flag = True


    def set_position(self, position):
        self.reader.set_position(position)
        self.last_position = position


    def read(self):
        if self.last_read_time is None: raise Exception('video is not started')

        frame, position = None, None

        while True:
            frame, position = self.reader.read()
            if frame is None: break

            current_time     = RealTimeVideoReader.get_time()
            current_position = position

            time_distance     = current_time - self.last_read_time
            position_distance = current_position - self.last_position

            if position_distance >= (time_distance / self.speed):
                self.last_read_time = current_time
                self.last_position = current_position
                break

        return frame, position