# 3rd party 
import torch

def bdcn_loss2(inputs, targets, l_weight=1.1):
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
    def __init__(self, weights):
        self.weights = weights


    def __call__(self, preds, target):
        if len(preds) != len(self.weights):
            raise Exception(f'len of output({len(preds)}) != len of weights ({len(self.weights)}')
        
        return sum(
            [ bdcn_loss2(pred, target, w) for pred, w in zip(preds, self.weights)]
        )