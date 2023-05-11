# typing
from dataclasses import dataclass
# project
from ...option     import option_find_float_default, option_find_int_default, option_validate_allow_parameters
from ..config_info import ConfigBlock

@dataclass
class UpsampleLayer:
    stride : int

    scale  : float

    config_block : ConfigBlock


ALLOW_UPSAMPLE_PARAMS = set([
    'stride',
    'scale'
])

def parse_upsample(data : ConfigBlock) -> UpsampleLayer:
    option_validate_allow_parameters(data, ALLOW_UPSAMPLE_PARAMS)

    stride = option_find_int_default(data, 'stride', 2)

    scale = option_find_float_default(data, 'scale', 1.0)
    

    return UpsampleLayer(stride, scale, data)