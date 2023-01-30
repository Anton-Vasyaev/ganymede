# python
from typing import cast
# 3rd party
import torch
import torch.nn as nn
# project
from .data.darknet_forward_size    import DarknetForwardSize
from .data.darknet_forward_type    import DarknetForwardType
from ...parsing.data.params import UpsampleLayer
from .darknet_layer_module         import DarknetLayerModule


class UpsampleModule(DarknetLayerModule):
#region data
    params : UpsampleLayer

    input_module : DarknetLayerModule
#endregion


#region construct_and_destruct
    def __init__(
        self, 
        params       : UpsampleLayer,
        input_module : DarknetLayerModule
    ):
        self.params       = params
        self.input_module = input_module

        input_size = input_module.output_size

        output_size = DarknetForwardSize(
            input_size.width * params.stride,
            input_size.height * params.stride,
            input_size.channels
        )

        DarknetLayerModule.__init__(self, input_size, output_size)

        self.nn_module = nn.Sequential(
            nn.Upsample(scale_factor=params.stride, mode='nearest')
        )
#endregion


#region interface_implementation
    def get_forward_type(self) -> DarknetForwardType:
        return DarknetForwardType.STANDARD

    
    def process(self) -> None:
        nn_module_c = cast(torch.nn.Module, self.nn_module)

        in_x = self.input_module.output

        out_x : torch.Tensor = nn_module_c(in_x)
        out_x = out_x * self.params.scale

        self.output = out_x
#endregion