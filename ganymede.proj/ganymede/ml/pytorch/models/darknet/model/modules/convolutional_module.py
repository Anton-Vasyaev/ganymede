# python
from typing import List, cast
# 3rd party
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter as nn_Parameter
from ganymede.rw import BinaryPackageReader
# project
from ...parsing.data.params import ConvolutionalLayer
from .data.darknet_forward_size import DarknetForwardSize
from .data.darknet_forward_type import DarknetForwardType
from ..activation import load_activation
from .darknet_layer_module import DarknetLayerModule

from .auxiliary import read_tensor_from_binary


def calculate_conv_output_dim(in_dim : int, kernel : int, stride : int, padding : int) -> int:
    return (in_dim + 2 * padding - kernel) // stride + 1


class ConvolutionalModule(DarknetLayerModule):
#region data
    params : ConvolutionalLayer

    input_module : DarknetLayerModule

    bias : nn_Parameter

    scales : nn_Parameter
#endregion


#region construct_and_destruct
    def __init__(
        self,
        params       : ConvolutionalLayer, 
        input_module : DarknetLayerModule
    ):
        self.params       = params
        self.input_module = input_module

        # set input and outputs information
        input_size  = input_module.output_size
        in_channels = input_size.channels

        out_channels    = params.filters
        out_height      = calculate_conv_output_dim(
            input_size.height, 
            params.size, 
            params.stride_y, 
            params.padding
        )
        out_width = calculate_conv_output_dim(
            input_size.width,
            params.size,
            params.stride_x,
            params.padding
        )

        output_size = DarknetForwardSize(out_width, out_height, out_channels)
        DarknetLayerModule.__init__(self, input_size, output_size)

        # build layers
        layers : List[nn.Module] = []

        bias_flag = True if params.batch_normalize == 0 else False
        layers.append(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                params.size, 
                (params.stride_y, params.stride_x),
                params.padding,
                bias=bias_flag
            )
        )

        if params.batch_normalize == 0:
            pass
        elif params.batch_normalize == 1:
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            raise Exception(f'Not implemented Conv2d for batch_normalize={params.batch_normalize}.')

        layers.append(load_activation(params.activation, params.config_block))

        self.nn_module = nn.Sequential(*layers)
#endregion


#region interface_implementation
    def get_forward_type(self) -> DarknetForwardType:
        return DarknetForwardType.STANDARD

    
    def process(self) -> None:
        in_x = self.input_module.output

        nn_module_c = cast(torch.nn.Module, self.nn_module)

        out_x : torch.Tensor = nn_module_c(in_x)

        self.output = out_x
#endregion

#region overriding

    def load_weights(self, reader : BinaryPackageReader):
        in_channels  = self.input_size.channels
        out_channels = self.params.filters
        kernel_size  = self.params.size

        # get layers
        nn_seq     = cast(nn.Sequential, self.nn_module)
        conv_layer = cast(nn.Conv2d, nn_seq[0])

        device = next(conv_layer.parameters()).device

        # read biases
        biases_t = read_tensor_from_binary(reader, (out_channels, ), device)

        # read batchnorm2d
        if self.params.batch_normalize == 1:
            batchnorm_layer = cast(nn.BatchNorm2d, nn_seq[1])

            # Set gamma
            scales_t               = read_tensor_from_binary(reader, (out_channels, ), device)
            batchnorm_layer.weight = nn_Parameter(scales_t, requires_grad=True)

            # Set beta
            batchnorm_layer.bias = nn_Parameter(biases_t, requires_grad=True)

            # Set running mean
            rolling_mean_t               = read_tensor_from_binary(reader, (out_channels, ), device)
            batchnorm_layer.running_mean = nn_Parameter(rolling_mean_t, requires_grad=False)

            # Set running variacne
            rolling_variance_t          = read_tensor_from_binary(reader, (out_channels, ), device)
            batchnorm_layer.running_var = nn_Parameter(rolling_variance_t, requires_grad=False)

        # set biases to conv2d if batchnorm is 0
        else:
            conv_layer.bias = nn_Parameter(biases_t, requires_grad=True)

        # read conv2d weights
        conv_weights_t = read_tensor_from_binary(
            reader, 
            #(in_channels, out_channels, kernel_size, kernel_size),
            (out_channels, in_channels, kernel_size, kernel_size),  
            device
        )
        # convert (channels, filters, size, size) -> (filters, channels, size, size)
        #conv_weights_t = conv_weights_t.permute(1, 0, 2, 3)

        conv_layer.weight = nn_Parameter(conv_weights_t, requires_grad=True)

#endregion
