# python
from enum import Enum, auto
from typing import Dict, Any, cast


class DarknetLayerType(Enum):
    CONVOLUTIONAL = auto()
    DECONVOLUTIONAL = auto()
    CONNECTED = auto()
    MAXPOOL = auto()
    LOCAL_AVGPOOL = auto()
    SOFTMAX = auto()
    DETECTION = auto()
    DROPOUT = auto()
    CROP = auto()
    ROUTE = auto()
    COST = auto()
    NORMALIZATION = auto()
    AVGPOOL = auto()
    LOCAL = auto()
    SHORTCUT = auto()
    SCALE_CHANNELS = auto()
    SAM = auto()
    ACTIVE = auto()
    RNN = auto()
    GRU = auto()
    LSTM = auto()
    CONV_LSTM = auto()
    HISTORY = auto()
    CRNN = auto()
    BATCHNORM = auto()
    NETWORK = auto()
    XNOR = auto()
    REGION = auto()
    YOLO = auto()
    GAUSSIAN_YOLO = auto()
    ISEG = auto()
    REORG = auto()
    REORG_OLD = auto()
    UPSAMPLE = auto()
    LOGXENT = auto()
    L2NORM = auto()
    EMPTY = auto()
    BLANK = auto()
    CONTRASTIVE = auto()
    IMPLICIT = auto()

    @staticmethod
    def from_str(data : str):
        present = DARKNET_LAYER_TYPE_STR_PRESENT_DICT

        if not data in present:
            raise ValueError(f'Cannot parse layer type:{data}')

        return DarknetLayerType(present[data])


    @staticmethod
    def contain_str_present(data : str) -> bool:
        return data in DARKNET_LAYER_TYPE_STR_PRESENT_DICT


DARKNET_LAYER_TYPE_STR_PRESENT_DICT = {
    'shortcut' : DarknetLayerType.SHORTCUT,
    'scale_channels': DarknetLayerType.SCALE_CHANNELS,
    'sam': DarknetLayerType.SAM,
    'crop': DarknetLayerType.CROP,
    'cost': DarknetLayerType.COST,
    'detection': DarknetLayerType.DETECTION,
    'region': DarknetLayerType.REGION,
    'yolo': DarknetLayerType.YOLO,
    'Gaussian_yolo': DarknetLayerType.GAUSSIAN_YOLO,
    'local': DarknetLayerType.LOCAL,
    'conv': DarknetLayerType.CONVOLUTIONAL,
    'convolutional': DarknetLayerType.CONVOLUTIONAL,
    'activation': DarknetLayerType.ACTIVE,
    'net': DarknetLayerType.NETWORK, 
    'network': DarknetLayerType.NETWORK,
    'crnn': DarknetLayerType.CRNN,
    'gru': DarknetLayerType.GRU,
    'lstm': DarknetLayerType.LSTM,
    'conv_lstm': DarknetLayerType.CONV_LSTM,
    'history': DarknetLayerType.HISTORY,
    'rnn': DarknetLayerType.RNN,
    'conn': DarknetLayerType.CONNECTED,
    'connected': DarknetLayerType.CONNECTED,
    'max': DarknetLayerType.MAXPOOL,
    'maxpool': DarknetLayerType.MAXPOOL,
    'local_avg': DarknetLayerType.LOCAL_AVGPOOL,
    'local_avgpool': DarknetLayerType.LOCAL_AVGPOOL,
    'reorg3d': DarknetLayerType.REORG,
    'reorg': DarknetLayerType.REORG_OLD,
    'avg': DarknetLayerType.AVGPOOL, 
    'avgpool': DarknetLayerType.AVGPOOL,
    'dropout': DarknetLayerType.DROPOUT,
    'lrn': DarknetLayerType.NORMALIZATION, 
    'normalization': DarknetLayerType.NORMALIZATION,
    'batchnorm': DarknetLayerType.BATCHNORM,
    'soft': DarknetLayerType.SOFTMAX, 
    'softmax': DarknetLayerType.SOFTMAX,
    'contrastive': DarknetLayerType.CONTRASTIVE,
    'route': DarknetLayerType.ROUTE,
    'upsample': DarknetLayerType.UPSAMPLE,
    'empty': DarknetLayerType.EMPTY,
    'silence': DarknetLayerType.EMPTY,
    'implicit': DarknetLayerType.IMPLICIT
}
