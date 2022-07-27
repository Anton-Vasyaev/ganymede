# 3rd party
import torch
import torch.nn as nn

class OneClassSegmentation:
    def __init__(self, threshold):
        self.sigmoid   = nn.Sigmoid()
        self.threshold = threshold


    def __call__(self, output):
        output  = self.sigmoid(output)
        bin_map = (output > self.threshold).type(torch.float32)
        return bin_map