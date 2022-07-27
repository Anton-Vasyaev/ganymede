import torch


class DexiNedFuseEdgeDetection:
    @staticmethod
    def normalization(
        output,
        epsilon=1e-12
    ):
        min_o = output.min()
        max_o = output.max()

        return (output - min_o) / (max_o - min_o + epsilon)


    def __init__(self, threshold):
        self.threshold = threshold
        self.sigmoid   = torch.nn.Sigmoid()

    
    def __call__(self, outputs):
        output = outputs[-1]
        output = self.sigmoid(output)
        output = self.normalization(output)

        bin_map = (output > self.threshold).type(torch.float32)
        return bin_map