import torch

class OneClassDiceLoss:
    def __init__(self):
        pass


    def forward(self, input : torch.Tensor, target : torch.Tensor):
        dice_coeff = 2 * (input * target).sum() / (input.sum() + target.sum() + 1e-6)
        loss       = 1 - dice_coeff

        return loss


    def __call__(self, input : torch.Tensor, target : torch.Tensor):
        return self.forward(input, target)