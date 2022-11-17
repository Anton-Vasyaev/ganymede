import dependencies
# 3rd party
import cv2 as cv
import ganymede.opencv as g_cv


def aug_nearest_downsize(img, width_scale, height_scale):
    orig_h, orig_w = img.shape[0:2]

    down_w, down_h = int(orig_w * width_scale), int(orig_h * height_scale)

    img = cv.resize(img, (down_w, down_h), interpolation=cv.INTER_NEAREST)

    img = cv.resize(img, (orig_w, orig_h), interpolation=cv.INTER_AREA)

    return img


if __name__ == '__main__':
    img = g_cv.imread('./../resources/images/tank.jpg')

    g_cv.imshow('debug', img)

    down_scale = 0.8

    while True:
        aug_img = aug_nearest_downsize(img, down_scale, 1.0)
        key = g_cv.imshow('debug', aug_img)

        if key == 97:
            down_scale = max(0.05, down_scale - 0.005)
        elif key == 100:
            down_scale = min(1.0, down_scale + 0.005)

        print(down_scale)

