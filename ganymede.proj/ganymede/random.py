# python
import random
from enum   import Enum
from random import Random
from typing import TypeVar, List, Type, Any

T     = TypeVar('T')
EnumT = TypeVar('T', bound=Enum)


def provide_default_instance_if_none(random_instance : Random) -> Random:
    if random_instance is None: return Random(0)
    else: return random_instance


def get_random_distance(
    start           : float  = 0.0, 
    end             : float  = 1.0,
    random_instance : Random = None
) -> float:
    random_instance = provide_default_instance_if_none(random_instance)

    distance = end - start

    rand_val = start + (random_instance.random() * distance)

    return rand_val




def get_random_bool(
    true_border     : float  = 0.5,
    random_instance : Random = None
) -> bool:
    random_instance = provide_default_instance_if_none(random_instance)

    val = random_instance.random()
    if val > true_border: return True
    else: return False


def get_random_enum(
    enum_type : Type[EnumT],
    random_instance = None
) -> EnumT:
    random_instance = provide_default_instance_if_none(random_instance)

    return enum_type(int(get_random_distance(0, len(enum_type))))


def choice(
    data : List[T],
    random_instance = None
) -> T:
    random_instance = provide_default_instance_if_none(random_instance)

    return random_instance.choice(data)


def sample(
    data            : List[T],
    len             : int,
    random_instance : Random = None
) -> List[T]:
    random_instance = provide_default_instance_if_none(random_instance)

    return random_instance.sample(data, len)


def multisample(
    data            : List[T],
    sample_len      : int,
    random_instance : random.Random = None
) -> List[T]:
    if len(data) < 1:
        raise ValueError(f'invalid len(data) < 0:{len(data)}')
    sample_list : List[T] = []

    current_len = sample_len
    data_len    = len(data)
    while current_len > 0:
        current_sample_len = min(data_len, current_len)
        sample_list += sample(data, current_sample_len, random_instance)
        current_len -= data_len

    return sample_list