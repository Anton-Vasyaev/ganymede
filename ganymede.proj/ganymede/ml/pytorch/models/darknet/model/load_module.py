# typing
from typing import Any
# project
from .modules import *
from .modules.data.darknet_forward_size import DarknetForwardSize
from .modules.darknet_layer_module import DarknetLayerModule
from ..parsing.data.params import *


STANDARD_FORWARD_BLOCK_PARAMS = set([
    ConvolutionalLayer,
    MaxPoolLayer,
    UpsampleLayer,
    YoloLayer
])


MULTINODE_FORWARD_BLOCK_PARAMS = set([
    RouteLayer
])


def load_standard_forward_module(
    layer_block  : Any, 
    input_module : DarknetLayerModule
) -> DarknetLayerModule:
    block_type = type(layer_block)

    if block_type is ConvolutionalLayer:
        return ConvolutionalModule(layer_block, input_module)
    elif block_type is MaxPoolLayer:
        return MaxPoolModule(layer_block, input_module)
    elif block_type is UpsampleLayer:
        return UpsampleModule(layer_block, input_module)
    elif block_type is YoloLayer:
        return YoloModule(layer_block, input_module)
    else:
        raise ValueError(f'Invalid params block type of standard forward module:{block_type}.')


def load_multinode_forward_module(
    layer_block     : Any,
    darknet_modules : List[DarknetLayerModule]
) -> DarknetLayerModule:
    block_type = type(layer_block)

    if block_type is RouteLayer:
        return RouteModule(layer_block, darknet_modules)
    else:
        raise ValueError(f'Invalid params block type of multinode forward module:{block_type}.')