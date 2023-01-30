# 3rd party
import torch
import torch.nn as nn
# project
from .data.darknet_forward_size    import DarknetForwardSize
from .data.darknet_forward_type    import DarknetForwardType
from .darknet_layer_module         import DarknetLayerModule
from ...parsing.data.params import YoloLayer


class YoloModule(DarknetLayerModule):
    params : YoloLayer

    input_module : DarknetLayerModule

    def __init__(
        self,
        params       : YoloLayer,
        input_module : DarknetLayerModule
    ):
        self.params       = params
        self.input_module = input_module

        input_size  = input_module.output_size
        output_size = input_size
        DarknetLayerModule.__init__(self, input_size, output_size)

        self.params = params


    def get_forward_type(self) -> DarknetForwardType:
        return DarknetForwardType.STANDARD

    
    def process(self) -> None:
        self.output = self.input_module.output