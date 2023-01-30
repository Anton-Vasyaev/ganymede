# python
from typing import Optional
from abc    import abstractmethod
# 3rd party
import torch
from ganymede.rw import BinaryPackageReader
# project
from .data.darknet_forward_size import DarknetForwardSize
from .data.darknet_forward_type import DarknetForwardType


class DarknetLayerModule:
    input_size : DarknetForwardSize

    output_size : DarknetForwardSize

    nn_module : Optional[torch.nn.Module]

    output : torch.Tensor

    def __init__(
        self,
        input_size  : DarknetForwardSize,
        output_size : DarknetForwardSize
    ):
        self.input_size  = input_size
        self.output_size = output_size

        self.nn_module = None

        self.output = torch.Tensor()


    @abstractmethod
    def get_forward_type(self) -> DarknetForwardType:
        raise NotImplementedError()

    
    @abstractmethod
    def process(self) -> None:
        raise NotImplementedError()

    
    # if layer is not contain weights no need to override this method
    def load_weights(self, reader : BinaryPackageReader):
        pass

