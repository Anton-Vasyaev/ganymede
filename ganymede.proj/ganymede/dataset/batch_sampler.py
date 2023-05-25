# python
import random
import concurrent.futures
import multiprocessing
from typing import Generic, TypeVar, Tuple, List
from concurrent.futures import ThreadPoolExecutor
# 3rd party
import cv2 as cv
import numpy as np
# project
from ganymede.imaging import ImageType
import ganymede.ml.pytorch.tensor as g_tensor

from .i_dataset_loader import IDatasetLoader


InputT  = TypeVar('InputT')
TargetT = TypeVar('TargetT')



def load_img_and_target_thread_func(
    dataset_loader : IDatasetLoader[InputT, TargetT], 
    idx            : int
) -> Tuple[InputT, TargetT]:
    input, target = dataset_loader[idx]

    return input, target


class BatchSampler(Generic[InputT, TargetT]):
    __dataset_loader : IDatasetLoader[InputT, TargetT]

    __batch_size : int

    __max_workers : int

    __indices : List[int]

    __executor : ThreadPoolExecutor

    __random_instance : random.Random

    __size : int


    def __init__(
        self, 
        dataset_loader : IDatasetLoader[InputT, TargetT], 
        batch_size     : int,
        threads        : int = 0,
        random_seed    : int = 1024
    ):
        self.__dataset_loader = dataset_loader
        self.__batch_size     = batch_size

        self.__max_workers = multiprocessing.cpu_count() if threads == 0 else threads
        
        self.__indices = np.arange(len(dataset_loader)).tolist()

        self.__executor = None
        if self.__max_workers != 1:
            self.__executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.__max_workers)

        self.__random_instance = random.Random(random_seed)

        data_len  = len(self.__dataset_loader)
        self.__size = data_len // self.__batch_size
        if data_len % self.__batch_size != 0: self.__size += 1


    def shuffle(self): self.__random_instance.shuffle(self.__indices)


    def __len__(self) -> int: return self.__size


    def __getitem__(self, idx : int) -> Tuple[List[InputT], List[TargetT]]:
        if idx < 0 or idx >= self.__size:
            raise Exception(f'invalid idx:{idx}, size:{self.__size}')

        start = idx * self.__batch_size
        end   = min(start + self.__batch_size, len(self.__dataset_loader))

        selected_indices = self.__indices[start:end]

        if self.__max_workers == 1:
            return self.current_thread_getitem(selected_indices)
        else:
            return self.threadpool_getitem(selected_indices)


    def current_thread_getitem(self, indices : List[int]) -> Tuple[InputT, TargetT]:
        images_list = []
        target_list = []

        for idx in indices:
            img, target = load_img_and_target_thread_func(self.__dataset_loader, idx)

            images_list.append(img)
            target_list.append(target)

        return images_list, target_list
        

    def threadpool_getitem(self, indices):
        images_list = []
        target_list = []

        future_to_img = {
            self.__executor.submit(
                load_img_and_target_thread_func, 
                self.__dataset_loader, 
                ind,
            ) : ind for ind in indices
        }

        for future in concurrent.futures.as_completed(future_to_img):
            img, target = future.result()

            images_list.append(img)
            target_list.append(target)

        return images_list, target_list