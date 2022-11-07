# python
from typing import List
# 3rd party
import torch


class DexiNedFuseEdgeDetection:
    threshold : float

    @staticmethod
    def normalization(
        output : torch.Tensor,
        epsilon : float = 1e-12
    ) -> torch.Tensor:
        min_o = output.min()
        max_o = output.max()

        return (output - min_o) / (max_o - min_o + epsilon)


    def __init__(
        self, 
        threshold : float
    ):
        self.threshold = threshold

    
    def __call__(self, outputs : List[torch.Tensor]) -> torch.Tensor:
        output = outputs[-1]
        output = torch.sigmoid(output)
        output = self.normalization(output)

        bin_map = (output > self.threshold).type(torch.float32)
        return bin_map