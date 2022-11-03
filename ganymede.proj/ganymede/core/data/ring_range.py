from __future__ import annotations
# 3rd party
from nameof import nameof
# project
from ..arithmetic import ring_add


class RingRangeIterator:
    def __init__(
        self, 
        start   : int,
        end     : int, 
        length  : int,
        reverse : bool = False
    ):
        validate_status = True
        if start < 0 or start >= length:
            validate_status = False
        if end < 0 or end > length:
            validate_status = False

        if not validate_status:
            raise ValueError(
                f'Invalid parameters for {nameof(RingRangeIterator)}:'
                f'start ({start}), end({end}), length({length}).'
            )

        self.cursor = start
        self.end    = end
        self.length = length

        if not reverse:
            self.move_value = 1
        else:
            self.move_value = -1
    

    def __iter__(self) -> RingRangeIterator:
        return self


    def __next__(self) -> int:
        val = self.cursor

        self.cursor = ring_add(self.cursor, self.move_value, self.length)

        if val == self.end:
            raise StopIteration()



def ring_range(
    start   : int, 
    end     : int,
    length  : int,
    reverse : bool = False
):
    return RingRangeIterator(start, end, length, reverse)