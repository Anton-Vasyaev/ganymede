import torch

class MSEKeypoint:
    def __init__(self):
        self.loss    = torch.nn.MSELoss()
        self.sigmoid = torch.nn.Sigmoid()

    
    def __call__(self, output, target):
        batch_size = output.shape[0]
        keypoints  = output.shape[1]

        output = output.view(batch_size, keypoints * 2)
        target = target.view(batch_size, keypoints * 2)

        output = self.sigmoid(output)

        return self.loss(output, target)