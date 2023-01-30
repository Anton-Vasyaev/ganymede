# 3rd party
import torch
# project
from .data.darknet_forward_size import DarknetForwardSize
from .data.darknet_forward_type import DarknetForwardType
from .darknet_layer_module import DarknetLayerModule

class InputModule(DarknetLayerModule):
    def __init__(
        self,
        width    : int,
        height   : int,
        channels : int
    ):
        input_size  = DarknetForwardSize(width, height, channels)
        output_size = input_size

        DarknetLayerModule.__init__(self, input_size, output_size)


#region interface_implementation

    def get_forward_type(self) -> DarknetForwardType:
        return DarknetForwardType.STANDARD

    
    def process(self) -> None:
        pass

#endregion


#region methods

    def set_input(self, in_x : torch.Tensor):
        self.__validate_input(in_x)

        self.output = in_x

#endregion


#region private_methods

    def __validate_input(self, in_x : torch.Tensor):
        if len(in_x.shape) != 4:
            raise ValueError(
                f'Invalid dimensions count in Darknet input:'
                f'{len(in_x.shape)} (expected 4).'
            ) 

        b, c, h, w = in_x.shape

        in_c = self.input_size.channels
        in_h = self.input_size.height
        in_w = self.input_size.width

        if c != in_c or h != in_h or w != in_w:
            raise ValueError(
                f'Size of input tensor != size of darknet model input:'
                f'({c}, {h}, {w}) != ({in_c}, {in_h}, {in_w}) (channels, height, width).'
            )

#endregion