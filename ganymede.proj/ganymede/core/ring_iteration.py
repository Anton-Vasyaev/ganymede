from typing import List



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
        
        
class RingListIterator[T]:
    __data : List[T]
    
    __iterator : RingIterator
    
    def __init__(self, data : List[T]):
        if len(data) == 0:
            raise ValueError('len of data is 0')
        self.__data = data
        
        self.__iterator = RingIterator(len(data))
        
        
    def __len__(self) -> int:
        return len(self.__data)
    
    
    def current_idx(self) -> int:
        return self.__iterator.current()
    
    
    def current(self) -> T:
        idx = self.__iterator.current()
        return self.__data[idx]
    
    
    def move(self, move_val : int):
        self.__iterator.move(move_val)