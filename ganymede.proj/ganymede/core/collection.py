from turtle import right
from typing import TypeVar, Tuple, List

T = TypeVar('T')


def relative_split_list(data : List[T], split_val : float) -> Tuple[List[T], List[T]]:
    split_idx = int(len(data) * split_val)

    left_data = data[:split_idx]
    right_data = data[split_idx:]

    return left_data, right_data
