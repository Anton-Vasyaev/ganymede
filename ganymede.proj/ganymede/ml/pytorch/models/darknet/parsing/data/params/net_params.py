# python
from dataclasses import dataclass
from typing      import List
# 3rd party
from nameof import nameof
# project
from ..enum_types.learning_rate_policy_type import LearningRatePolicyType
from ..config_info import ConfigBlock
from ...option import *


@dataclass
class NetParams:
    batch : int

    subdivision : int

    width : int

    height : int

    channels : int

    momentum : float

    decay : float

    angle : float

    saturation : float

    exposure : float

    hue : float

    learning_rate : float

    burn_in : int

    max_batches : int

    policy : LearningRatePolicyType

    steps : List[int]

    scales : List[float]

    config_block : ConfigBlock


ALLOW_NET_PARAMS = set([
    'batch',
    'subdivision',
    'width',
    'height',
    'channels',
    'momentum',
    'decay',
    'angle',
    'saturation',
    'exposure',
    'hue',
    'learning_rate',
    'burn_in',
    'max_batches',
    'policy',
    'steps',
    'scales'
])

def parse_net(data : ConfigBlock) -> NetParams:
    max_batches = option_find_int_default(data, 'max_batches', 0)
    batch       = option_find_int_default(data, 'batch', 1)
    
    learning_rate = option_find_float_default(data, 'learning_rate', 1e-3)
    decay         = option_find_float_default(data, 'decay', 1e-4)
    momentum      = option_find_float_default(data, 'momentum', 0.9)

    subdivision = option_find_int_default(data, 'subdivision', 1)

    height   = option_find_int(data, 'height')
    width    = option_find_int(data, 'width')
    channels = option_find_int(data, 'channels')

    angle      = option_find_float_default(data, 'angle', 1.0)
    saturation = option_find_float_default(data, 'saturation', 1.0)
    exposure   = option_find_float_default(data, 'exposure', 1.0)
    hue        = option_find_float_default(data, 'hue', 1.0)
    
    burn_in = option_find_int_default(data, 'burn_in', 0)

    policy = LearningRatePolicyType.from_str(option_find_str_default(data, 'policy', 'constant'))

    steps = option_find_int_list(data, 'steps')

    scales = option_find_float_list(data, 'scales')

    return NetParams(
        batch,
        subdivision,
        width,
        height,
        channels,
        momentum,
        decay,
        angle,
        saturation,
        exposure,
        hue,
        learning_rate,
        burn_in,
        max_batches,
        policy,
        steps,
        scales,
        data
    )

