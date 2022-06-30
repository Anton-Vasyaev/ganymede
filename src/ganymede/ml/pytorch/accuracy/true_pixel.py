# 3rd party
import torch

class AccuracyTruePixel:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    
    def __call__(self, output, target):
        output = (output > self.threshold)
        target = (target > 0.99)

        accuracy = ((output == target) & (output != False)).sum() / target.sum()

        return accuracy