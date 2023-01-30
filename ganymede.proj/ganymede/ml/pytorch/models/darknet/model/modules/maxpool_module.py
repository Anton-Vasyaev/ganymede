# python
from typing import cast
# 3rd party
import torch
import torch.nn as nn
# project
from ...parsing.data.params import MaxPoolLayer
from .data.darknet_forward_size import DarknetForwardSize
from .data.darknet_forward_type import DarknetForwardType
from .darknet_layer_module import DarknetLayerModule


def calculate_maxpool_output_dim(in_dim : int, kernel : int, stride : int, padding : int) -> int:
    return (in_dim + padding - kernel) // stride + 1


class MaxPoolModule(DarknetLayerModule):
#region data
    params : MaxPoolLayer

    input_module : DarknetLayerModule
#endregion


#region construct_and_destruct
    def __init__(
        self,
        params       : MaxPoolLayer,
        input_module : DarknetLayerModule
    ):
        self.params       = params
        self.input_module = input_module

        input_size = input_module.output_size

        in_w = input_size.width
        in_h = input_size.height
        in_c = input_size.channels

        out_w = calculate_maxpool_output_dim(in_w, params.size, params.stride_x, params.padding)
        out_h = calculate_maxpool_output_dim(in_h, params.size, params.stride_x, params.padding)
        output_size = DarknetForwardSize(out_w, out_h, in_c)

        DarknetLayerModule.__init__(self, input_size, output_size)

        # oriented for code in method forward_maxpool_layer, src file: maxpool_layer.c (line 283)
        # int w_offset = -l.pad / 2;
        # int h_offset = -l.pad / 2;
        real_padding_size = params.padding // 2

        self.nn_module = nn.Sequential(
            nn.MaxPool2d(params.size, (params.stride_y, params.stride_y), real_padding_size)
        )
#endregion

#region interface_implementation
    def get_forward_type(self) -> DarknetForwardType:
        return DarknetForwardType.STANDARD

    
    def process(self) -> None:
        nn_module_c = cast(torch.nn.Module, self.nn_module)

        in_x = self.input_module.output

        out_x : torch.Tensor = nn_module_c(in_x)

        self.output = out_x
#endregion