import typing
from abc import abstractmethod
from typing import TypeVar, Protocol


def ring_add(
    value : int, 
    add   : int, 
    size  : int
):
    value = (value + add) % size

    if value < 0:
        value = size - value

    return value




class RingIterator:
    __length : int
    
    __current_idx : int 
    
    def __init__(
        self,
        length    : int,
        start_idx : int = 0
    ):
        if length <= 0:
            raise Exception(f'invalid len of ring:{length}.')
        
        if start_idx < 0 or start_idx >= length:
            raise Exception(
                f'invalid start idx ({start_idx})' 
                f'for ring with length:{length}'
            )
            
        self.__length      = length
        self.__current_idx = start_idx
        
        
    def current(self) -> int:
        return self.__current_idx
    
    
    def length(self) -> int:
        return self.__length
    
    
    def move(self, move_val : int):
        self.__current_idx = ring_add(
            self.__current_idx,
            move_val,
            self.__length
        )