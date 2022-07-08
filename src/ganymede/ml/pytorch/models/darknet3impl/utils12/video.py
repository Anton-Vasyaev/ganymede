import cv2 as cv

def drop_frames(capture, minutes, seconds):
    start_timestamp = (minutes * 60 + seconds) * 1000
    while True:
        ret1, frame1 = capture.read()

        if not ret1:
            return

        msec = capture.get(cv.CAP_PROP_POS_MSEC)

        if msec < start_timestamp:
            msec = int(msec)
            minutes = int(msec / (60 * 1000))
            seconds = int(msec % (60 * 1000) / 1000)
            print(f'videos pos:{minutes}:{seconds}')
            continue
        
        return