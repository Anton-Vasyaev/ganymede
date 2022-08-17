# python
import random
import concurrent.futures
import multiprocessing
# 3rd party
import cv2 as cv
import numpy as np
# project
from ganymede.imaging import ImageType
import ganymede.ml.pytorch.tensor as g_tensor


def load_img_and_target_thread_func(
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

        self.max_workers = multiprocessing.cpu_count() if threads == 0 else threads

        self.executor = None
        if self.max_workers != 1:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        self.indices = np.arange(len(dataset_loader)).tolist()

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

        if self.max_workers == 1:
            return self.current_thread_getitem(selected_indices)
        else:
            return self.threadpool_getitem(selected_indices)


    def current_thread_getitem(self, indices):
        images_list = []
        target_list = []

        for idx in indices:
            img, target = load_img_and_target_thread_func(self.dataset_loader, idx)

            images_list.append(img)
            target_list.append(target)

        return images_list, target_list
        

    def threadpool_getitem(self, indices):
        images_list = []
        target_list = []

        future_to_img = {
            self.executor.submit(
                load_img_and_target_thread_func, 
                self.dataset_loader, 
                ind,
            ) : ind for ind in indices
        }

        for future in concurrent.futures.as_completed(future_to_img):
            img, target = future.result()

            images_list.append(img)
            target_list.append(target)

        return images_list, target_list