# python
from typing import List, cast
# 3rd party 
import torch

def bdcn_loss2(
    inputs : torch.Tensor, 
    targets : torch.Tensor, 
    l_weight : float =1.1
) -> torch.Tensor:
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0  * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum

    return l_weight * cost


class BDCNLossCriterion:
    weights : List[float]


    def __init__(
        self, 
        weights : List[float]
    ):
        self.weights = weights


    def __call__(
        self, 
        preds  : torch.Tensor, 
        target : torch.Tensor
    ) -> torch.Tensor:
        if len(preds) != len(self.weights):
            raise Exception(f'len of output({len(preds)}) != len of weights ({len(self.weights)}')
        
        sum_result = sum(
            [ bdcn_loss2(pred, target, w) for pred, w in zip(preds, self.weights)]
        ) / len(preds)

        return cast(torch.Tensor, sum_result)