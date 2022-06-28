# python
import random
import concurrent.futures
import multiprocessing
# 3rd party
import cv2 as cv
import numpy as np
# project
from ganymede.imaging.image import ImageType
import ganymede.ml.pytorch.tensor as g_tensor


def __load_img_and_target(
    dataset_loader, 
    idx
):
    img, target = dataset_loader[idx]

    return img, target


class BatchSampler:
    def __init__(
        self, 
        dataset_loader, 
        batch_size,
        random_seed = 1024, 
        threads     = 0
    ):
        self.random_i = random.Random(random_seed)

        max_workers = multiprocessing.cpg_count() if threads == 0 else threads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self.indices = list(np.arange(len(dataset_loader)))

        self.dataset_loader = dataset_loader
        self.batch_size     = batch_size

        data_len  = len(self.dataset_loader)
        self.size = data_len // self.batch_size
        if data_len % self.batch_size != 0: self.size += 1


    def shuffle(self): self.random_i.shuffle(self.indices)


    def __len__(self): return self.size


    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise Exception(f'invalid idx:{idx}, size:{self.size}')

        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self.dataset_loader))

        selected_indices = self.indices[start:end]

        images_list = []
        target_list = []

        future_to_img = {
            self.executor.submit(
                __load_img_and_target, 
                self.dataset_loader, 
                ind,
            ) : ind for ind in selected_indices
        }

        for future in concurrent.futures.as_completed(future_to_img):
            img, target = future.result()

            images_list.append(img)
            target_list.append(target)

        return images_list, target_list