# python
from typing import List, Optional
# 3rd party
import torch
import torch.nn as nn
# project
from .data.darknet_forward_size import DarknetForwardSize
from .data.darknet_forward_type import DarknetForwardType
from ...parsing.data.params import RouteLayer
from ...parsing.error import DarknetConfigLoadException
from .darknet_layer_module import DarknetLayerModule


class RouteModule(DarknetLayerModule):
#region data
    params : RouteLayer

    input_module : List[DarknetLayerModule]
#endregion

#region construct_and_destruct
    def __init__(
        self, 
        params          : RouteLayer,
        darknet_modules : List[DarknetLayerModule]
    ):
        self.params = params
        
        in_size = DarknetForwardSize(-1, -1, -1)

        in_channels = 0
        out_channels = 0

        input_modules : List[DarknetLayerModule] = list()

        for layer_idx in params.layers:
            abs_layer_idx = len(darknet_modules) + layer_idx if layer_idx < 0 else layer_idx


            if abs_layer_idx < 0 or abs_layer_idx >= len(darknet_modules):
                raise DarknetConfigLoadException(
                    f'Invalid idx in route block: {layer_idx} (abs value:{abs_layer_idx})',
                    params.config_block.get_file_path(),
                    params.config_block.line_number
                )

            curr_module = darknet_modules[abs_layer_idx]

            if in_size.channels == -1:
                in_size = curr_module.output_size

            if curr_module.output_size != in_size:
                raise DarknetConfigLoadException(
                    f'Not equal sizes of inputs in route block:'
                    f'{in_size} != {curr_module.output_size} (channels, height, width)',
                    params.config_block.get_file_path(),
                    params.config_block.line_number
                )

            if curr_module.output_size.channels % params.groups != 0:
                raise DarknetConfigLoadException(
                    f'Channels in one input (layer idx:{layer_idx}) of route block '
                    f'is not multiple of route groups ({params.groups})',
                    params.config_block.get_file_path(),
                    params.config_block.line_number
                )

            in_channels += curr_module.output_size.channels

            part_group_channels = curr_module.output_size.channels // params.groups

            out_channels += part_group_channels

            input_modules.append(curr_module)

        input_size = DarknetForwardSize(in_size.width, in_size.height, in_channels)
        output_size = DarknetForwardSize(in_size.width, in_size.height, out_channels)
        DarknetLayerModule.__init__(self, input_size, output_size)

        self.input_modules = input_modules
#endregion


#region interface_implementation
    def get_forward_type(self) -> DarknetForwardType:
        return DarknetForwardType.MULTINODE

    
    def process(self) -> None:
        cat_list : List[torch.Tensor] = []

        group_id = self.params.group_id

        for input_module in self.input_modules:
            input = input_module.output

            input_c = input.shape[1]

            part_size = input_c // self.params.groups

            start = group_id * part_size

            cat_list.append(input[:, start:start+part_size,...])
        
        out_x = torch.cat(cat_list, dim=1)

        self.output = out_x
#endregion