# 3rd party
import torch
import torch.nn as nn


class OneClassSegmentator:
    def __init__(self, threshold):
        self.sigmoid   = nn.Sigmoid()
        self.threshold = threshold


    def __call__(self, output):
        output = torch.sigmoid(output)

        bin_map = (output > self.threshold).type(torch.float32)

        return bin_map