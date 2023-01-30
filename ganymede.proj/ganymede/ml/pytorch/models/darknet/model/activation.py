# python
import torch.nn as nn
# project
from ..parsing.data.enum_types.activation_type import ActivationType
from ..parsing.data.config_info import ConfigBlock
from ..parsing.error import DarknetConfigLoadException

def load_activation(activation_type : ActivationType, block : ConfigBlock) -> nn.Module:
    if activation_type == ActivationType.RELU:
        return nn.ReLU()
    elif activation_type == ActivationType.LOGISTIC:
        return nn.Sigmoid()
    elif activation_type == ActivationType.LEAKY:
        return nn.LeakyReLU(0.1)
    elif activation_type == ActivationType.LINEAR:
        return nn.Identity()
    else:
        raise DarknetConfigLoadException(
            f'Not impelemented activation type:{activation_type} in block \'{block.name}\'',
            block.get_file_path(),
            block.line_number
        )