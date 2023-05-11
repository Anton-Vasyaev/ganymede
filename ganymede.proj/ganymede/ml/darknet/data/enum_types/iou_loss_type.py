# python
from enum import IntEnum, auto


class IoULossType(IntEnum):
    IOU = auto() 
    GIOU = auto() 
    MSE = auto() 
    DIOU = auto() 
    CIOU = auto()

    @staticmethod
    def from_str(data : str):
        if data == 'iou': return IoULossType.IOU
        elif data == "mse": return IoULossType.MSE
        elif data == 'giou': return IoULossType.GIOU
        elif data == 'diou': return IoULossType.DIOU
        elif data == 'ciou': return IoULossType.CIOU
        else:
            raise ValueError(f'Cannot parse intersection over union loss type:{data}')