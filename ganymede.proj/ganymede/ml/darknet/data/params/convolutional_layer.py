# python
from dataclasses import dataclass
# project
from ..enum_types.activation_type import ActivationType
from ...option import option_find_int_default, option_find_str_default, option_validate_allow_parameters
from ..config_info import ConfigBlock


@dataclass
class ConvolutionalLayer:
    activation : ActivationType
    
    filters : int

    size : int

    stride_x : int

    stride_y : int

    padding : int

    batch_normalize : int

    config_block : ConfigBlock


ALLOW_CONVOLUTIONAL_PARAMS = set([
    'filters',
    'size',
    'stride_x',
    'stride_y',
    'stride',
    'pad',
    'padding',
    'activation',
    'batch_normalize'
])


def parse_convolutional(data : ConfigBlock) -> ConvolutionalLayer:
    option_validate_allow_parameters(
        data,
        ALLOW_CONVOLUTIONAL_PARAMS
    )

    filters = option_find_int_default(data, 'filters', 1)

    size = option_find_int_default(data, 'size', 1)

    stride_x = option_find_int_default(data, 'stride_x', -1)
    stride_y = option_find_int_default(data, 'stride_y', -1)

    if(stride_x == -1 or stride_y == -1):
        stride = option_find_int_default(data, 'stride', 1)

        stride_x = stride_x if stride_x != -1 else stride
        stride_y = stride_y if stride_y != -1 else stride

    pad = option_find_int_default(data, 'pad', 0)
    padding = option_find_int_default(data, 'padding', 0)

    if pad != 0:
        padding = size // 2

    activation_str = option_find_str_default(data, "activation", "logistic")
    activation     = ActivationType.from_str(activation_str)

    batch_normalize = option_find_int_default(data, "batch_normalize", 0)

    return ConvolutionalLayer(
        activation, 
        filters, 
        size, 
        stride_x, 
        stride_y, 
        padding, 
        batch_normalize, 
        data
    )

    



