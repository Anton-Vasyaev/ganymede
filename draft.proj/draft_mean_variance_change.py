import dependencies
# 3rd party
import cv2 as cv
import numpy as np

import ganymede.opencv as g_cv

from nameof import nameof


def verify_not_in_range(value, min, max, argument_name : str = ''):
    if value < min or value > max:
        raise ValueError(f'Value \'{argument_name}\' not in range [{min}, {max}]: {value}.')



def aug_change_mean_variance(img : np.ndarray, aug_mean : float, aug_variance : float):
    verify_not_in_range(aug_mean, 0.0, 1.0, nameof(aug_mean))
    verify_not_in_range(aug_variance, 0.0, 1.0, nameof(aug_variance))


    min_val, max_val = 0.0, 255.0

    if np.issubdtype(img.dtype, np.floating):
        max_val = 1.0

    mean = img.mean()

    variance = img.var() / 255

    img = img - mean

    img /= variance

    img *= aug_variance

    img += aug_mean

    img = np.clip(img, min_val, max_val)

    return img


if __name__ == '__main__':
    img = g_cv.imread('./../resources/images/tank.jpg')

    g_cv.imshow('debug', img)

    aug_mean = img.mean() / 255
    aug_variance = img.var() / (255 * 255)

    print(f'aug var:{aug_variance}')

    while True:
        aug_img = aug_change_mean_variance(img, aug_mean, aug_variance)
        key = g_cv.imshow('debug', aug_img)

        if key == 97:
            aug_mean = max(0.0, aug_mean - 0.005)
        elif key == 100:
            aug_mean = min(1.0, aug_mean + 0.005)
        elif key == 122:
            aug_variance = max(0.0, aug_variance - 0.005)
        elif key == 99:
            aug_variance = min(1.0, aug_variance + 0.005)

        print(f'aug mean:{aug_mean}, aug variance:{aug_variance}')

