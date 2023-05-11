# python
from dataclasses import dataclass, field
# project
from ..config_info import ConfigBlock
from ...option import option_validate_allow_parameters, option_find_int_default


@dataclass
class MaxPoolLayer:
    stride_x : int

    stride_y : int

    size : int

    padding : int

    config_block : ConfigBlock = field(repr=False)


ALLOW_MAXPOOL_PARAMS = set([
    'stride',
    'stride_x',
    'stride_y',
    'size',
    'padding'
])


def parse_maxpool(data : ConfigBlock) -> MaxPoolLayer:
    option_validate_allow_parameters(data, ALLOW_MAXPOOL_PARAMS)

    stride = option_find_int_default(data, 'stride', 1)

    stride_x = option_find_int_default(data, 'stride_x', stride)
    stride_y = option_find_int_default(data, 'stride_y', stride)

    size = option_find_int_default(data, 'size', stride)

    padding = option_find_int_default(data, 'padding', size - 1)

    return MaxPoolLayer(stride_x, stride_y, size, padding, data)