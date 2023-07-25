from typing import Tuple, Collection, TypeVar


T = TypeVar('T')


def collection_min_max(data: Collection[T]) -> Tuple[T, T]:
    min_v, max_v = data[0], data[0]
    for element in data[1:]:
        min_v = min(min_v, element)
        max_v = max(max_v, element)

    return min_v, max_v