# python
import random
# 3rd party
from typing import Tuple
import numpy as np
# project
import ganymede.opencv as g_cv
import ganymede.random as g_rand
from ganymede.dataset.i_dataset_loader import IDatasetLoader
from ganymede.dataset.batch_sampler    import BatchSampler

from ganymede.math.primitives import Color3



class OwnDatasetLoader(IDatasetLoader[np.ndarray, Color3]):
    __range : int

    def __init__(self, range : int):
        self.__range = range


    def __len__(self) -> int:
        return self.__range
    

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Color3]:
        ri = random.Random(idx)

        r = g_rand.get_random_distance(0, 255, ri)
        g = g_rand.get_random_distance(0, 255, ri)
        b = g_rand.get_random_distance(0, 255, ri)

        r, g, b = int(r), int(g), int(b)

        img = np.zeros((600, 600, 3)) + (r, g, b)
        img = img.astype(np.uint8)

        return img, (r, g, b)



if __name__ == '__main__':
    dataset = OwnDatasetLoader(1024)

    batch_sampler = BatchSampler[np.ndarray, Color3](dataset, 8)

    for idx in range(len(batch_sampler)):
        img_sample, color_sample = batch_sampler[idx]

        for img, color in zip(img_sample, color_sample):
            print(color)
            g_cv.imshow('debug', img)

