# python
from turtle import right
from typing import Optional, Tuple, List, Collection, Deque
from typing import TypeVar, Generic, cast

from numbers import Number

from collections import deque

import random
# 3rd party
import numpy as np


T = TypeVar('T')

NumT = TypeVar('NumT', bound=Number)


def relative_split_list(data : List[T], split_val : float) -> Tuple[List[T], List[T]]:
    split_idx = int(len(data) * split_val)

    left_data = data[:split_idx]
    right_data = data[split_idx:]

    return left_data, right_data


def normalize_collection(data : Collection[NumT]) -> List[float]:
    sum = 0.0

    for el in data:
        sum += el

    normalize_list = []
    for el in data:
        normalize_list.append(el / sum)

    return normalize_list


class RandomIndexDistributor:
    __index_collection : List[int]

    __random_instance : random.Random

    __random_elements : Deque[int]

    def __reinitialize(self):
        self.__random_elements = deque(
            self.__random_instance.sample(
                self.__index_collection, 
                len(self.__index_collection)
            )
        )


    def __init__(
        self, 
        len : int,
        random_instance : Optional[random.Random]
    ):
        self.__index_collection = cast(List[int], np.arange(len).tolist())

        if not random_instance is None:
            self.__random_instance = random_instance
        else:
            self.__random_instance = cast(random.Random, random)  

        self.__random_elements = deque()


    def next_element(self) -> int:
        if len(self.__random_elements) == 0:
            self.__reinitialize()

        return self.__random_elements.popleft()

    