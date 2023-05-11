import typing
from abc import abstractmethod
from typing import TypeVar, Protocol


def ring_add(value, add, size):
    value = (value + add) % size

    if value < 0:
        value = size - value

    return value