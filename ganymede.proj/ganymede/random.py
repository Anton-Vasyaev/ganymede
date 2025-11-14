# python
import ganymede.core as g_core

import random
import numpy as np
from enum   import Enum
from random import Random
from typing import TypeVar, List, Sequence, Tuple, Type, Optional, cast

T     = TypeVar('T')
EnumT = TypeVar('EnumT', bound=Enum)


class RandomWrapper:
    __random_instance : random.Random

    def __init__(self, random_instance : random.Random):
        self.__random_instance = random_instance


    def rand_distance(self, start : float, end : float) -> float:
        distance = end - start

        rand_val = start + (self.__random_instance.random() * distance)

        return rand_val


    def rand_range(self, range : Tuple[float, float]) -> float:
        return self.rand_distance(
            range[0],
            range[1]
        )

    
    def rand_bool(self, prob : float) -> bool:
        val = self.__random_instance.random()
        return val <= prob


    def rand_enum(self, enum_type : Type[EnumT]) -> EnumT:
        return enum_type(int(self.__random_instance.randint(0, len(enum_type))))

    
    def rand_int_range(self, range : Tuple[int, int]) -> int:
        return self.__random_instance.randint(range[0], range[1])


    def multisample(
        self,
        data       : Sequence[T],
        sample_len : int
    ) -> List[T]:
        if len(data) < 1:
            raise ValueError(f'invalid len(data) < 0:{len(data)}')
        sample_list : List[T] = []

        current_len = sample_len
        data_len    = len(data)
        while current_len > 0:
            current_sample_len = min(data_len, current_len)
            sample_list += self.__random_instance.sample(
                data, 
                current_sample_len
            )
            current_len -= data_len

        return sample_list


DEFAULT_WRAPPER = RandomWrapper(cast(Random, random))

def rand_distance(
    start           : float  = 0.0, 
    end             : float  = 1.0
) -> float:
    return DEFAULT_WRAPPER.rand_distance(start, end)


def rand_range(
    range : Tuple[float, float],
):
    return DEFAULT_WRAPPER.rand_range(range)


def rand_bool(
    prob            : float  = 0.5,
) -> bool:
    return DEFAULT_WRAPPER.rand_bool(prob)


def rand_enum(
    enum_type       : Type[EnumT]
) -> EnumT:
    return DEFAULT_WRAPPER.rand_enum(enum_type)


def multisample(
    data            : Sequence[T],
    sample_len      : int,
) -> List[T]:
    return DEFAULT_WRAPPER.multisample(data, sample_len)


def choice_by_probs(
    items  : List[T],
    probs  : List[T]
) -> T:
    selected_item_a = np.random.choice(items, p=probs)

    return cast(T, selected_item_a) 


def choice_by_scores(
    items  : List[T],
    scores : List[T]
) -> T:
    probs = g_core.normalize_collection(scores)
    
    return choice_by_probs(items, probs)