# python
from enum import Enum, auto

class LearningRatePolicyType(Enum):
    CONSTANT = auto()
    STEP = auto() 
    EXP = auto() 
    POLY = auto() 
    STEPS = auto() 
    SIG = auto() 
    RANDOM = auto() 
    SGDR = auto()

    @staticmethod
    def from_str(data : str):
        if data == 'random': return LearningRatePolicyType.RANDOM
        elif data == 'poly': return LearningRatePolicyType.POLY
        elif data == 'constant': return LearningRatePolicyType.CONSTANT
        elif data == 'step': return LearningRatePolicyType.STEP
        elif data == 'exp': return LearningRatePolicyType.EXP
        elif data == 'sigmoid': return LearningRatePolicyType.SIG
        elif data == 'steps': return LearningRatePolicyType.STEPS
        elif data == 'sgdr': return LearningRatePolicyType.SGDR
        else: raise ValueError(f'Cannot parse learning rate policy type:{data}.')