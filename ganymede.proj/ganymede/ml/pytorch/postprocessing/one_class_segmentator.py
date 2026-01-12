# python
from typing import Callable
# 3rd party
import torch
import torch.nn as nn


class OneClassSegmentator:
    threshold : float 

    activation_function : Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self, 
        threshold           : float,
        activation_function : Callable[[torch.Tensor], torch.Tensor]
    ):
        self.threshold = threshold

        self.activation_function = activation_function


    def __call__(self, output):
        output = self.activation_function(output)

        bin_map = (output > self.threshold).type(torch.float32)

        return bin_map