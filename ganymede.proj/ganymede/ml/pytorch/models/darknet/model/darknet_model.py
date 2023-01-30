# python
from typing import List
# 3rd party
import torch
import torch.nn as nn
from ganymede.rw import BinaryPackageReader
# project
from ..parsing.parse_blocks     import *
from ..parsing.data.params      import *
from .modules.data.darknet_forward_size import DarknetForwardSize
from .modules import *
from .load_module import *

class DarknetModel(nn.Module):
#region static_methods
    @staticmethod
    def load_from_file(path : str):
        return DarknetModel(read_darknet_bone_from_file(path))


    @staticmethod
    def load_from_str(data : str):
        return DarknetModel(read_darknet_bone_from_str(data))

#endregion


#region data
    net_params : NetParams

    modules : List[DarknetLayerModule]

    input_module : InputModule

    seen : int

#endregion


#region construct_and_destruct
    def __init__(self, network_bone : DarknetNetworkBone):
        super().__init__()

        net_params = network_bone.net_params
        self.net_params = net_params

        self.input_module = InputModule(net_params.width, net_params.height, net_params.channels)
        prev_module = self.input_module

        nn_module_idx = 1
        self.modules = list()
        for layer_block in network_bone.layers:
            layer_type = type(layer_block)
            if layer_type in STANDARD_FORWARD_BLOCK_PARAMS:
                curr_module = load_standard_forward_module(layer_block, prev_module)
            elif layer_type in MULTINODE_FORWARD_BLOCK_PARAMS:
                curr_module = load_multinode_forward_module(layer_block, self.modules)
            else:
                raise Exception(f'Not implemeneted layer block:{type(layer_block)}')

            if not curr_module.nn_module is None:
                setattr(self, f'module_{nn_module_idx}', curr_module.nn_module)
                nn_module_idx += 1

            prev_module = curr_module

            self.modules.append(curr_module)
#endregion


#region methods
    def load_weights_from_file(self, path : str) -> None:
        with open(path, 'rb') as fh:
            data = fh.read()

            self.load_weights_from_binary(data)


    def load_weights_from_binary(self, data : bytes) -> None:
        reader = BinaryPackageReader(data)

        major = reader.read_int32()
        minor = reader.read_int32()
        revision = reader.read_int32()
        
        if major * 10 + minor >= 2:
            self.seen = reader.read_uint64()
        else:
            self.seen = reader.read_uint32()

        for module in self.modules:
            module.load_weights(reader)


    def get_output_modules(self) -> List[DarknetLayerModule]:
        output_modules : List[DarknetLayerModule] = list()

        for module in self.modules:
            if isinstance(module, YoloModule):
                output_modules.append(module)

        return output_modules


    def forward(self, in_x : torch.Tensor) -> List[torch.Tensor]:
        self.input_module.set_input(in_x)

        for module in self.modules:
            module.process()

        output_modules = self.get_output_modules()

        outputs : List[torch.Tensor] = list()

        for module in output_modules:
            outputs.append(module.output)

        return outputs
#endregion