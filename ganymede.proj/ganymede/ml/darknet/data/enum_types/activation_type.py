from enum import Enum, auto

class ActivationType(Enum):
    LOGISTIC = auto() 
    RELU = auto() 
    RELU6 = auto() 
    RELIE = auto() 
    LINEAR = auto() 
    RAMP = auto() 
    TANH = auto() 
    PLSE = auto() 
    REVLEAKY = auto() 
    LEAKY = auto() 
    ELU = auto() 
    LOGGY = auto() 
    STAIR = auto() 
    HARDTAN = auto() 
    LHTAN = auto() 
    SELU = auto() 
    GELU = auto() 
    SWISH = auto() 
    MISH = auto() 
    HARD_MISH = auto() 
    NORM_CHAN = auto() 
    NORM_CHAN_SOFTMAX = auto() 
    NORM_CHAN_SOFTMAX_MAXVAL = auto()
    

    @staticmethod
    def from_str(data : str):
        if data == 'logistic': return ActivationType.LOGISTIC
        elif data == 'swish': return ActivationType.SWISH
        elif data == 'mish': return ActivationType.MISH
        elif data == 'hard_mish': return ActivationType.HARD_MISH
        elif data == 'normalize_channels': return ActivationType.NORM_CHAN
        elif data == 'normalize_channels_softmax': return ActivationType.NORM_CHAN_SOFTMAX
        elif data == 'normalize_channels_softmax_maxval': return ActivationType.NORM_CHAN_SOFTMAX_MAXVAL
        elif data == 'loggy': return ActivationType.LOGGY
        elif data == 'relu': return ActivationType.RELU
        elif data == 'relu6': return ActivationType.RELU6
        elif data == 'elu': return ActivationType.ELU
        elif data == 'selu': return ActivationType.SELU
        elif data == 'gelu': return ActivationType.GELU
        elif data == 'relie': return ActivationType.RELIE
        elif data == 'plse': return ActivationType.PLSE
        elif data == 'hardtan': return ActivationType.HARDTAN
        elif data == 'lhtan': return ActivationType.LHTAN
        elif data == 'linear': return ActivationType.LINEAR
        elif data == 'ramp':return ActivationType.RAMP
        elif data == 'revleaky': return ActivationType.REVLEAKY
        elif data == 'leaky': return ActivationType.LEAKY
        elif data == 'tanh': return ActivationType.TANH
        elif data == 'stair': return ActivationType.STAIR
        else:
            raise ValueError(f'Cannot parse activation:{data}')
