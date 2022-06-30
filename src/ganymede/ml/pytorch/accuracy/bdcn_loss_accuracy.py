# project
from ganymede.ml.pytorch.loss.bdcn import BDCNLossCriterion, bdcn_loss2


class BDCNLossAccuracyFunctor:
    def __init__(
        self,
        weights
    ):
        self.loss_functor = BDCNLossCriterion(weights)
        self.weights = weights


    def __call__(self, preds, target):
        if len(preds) != len(self.weights):
            raise Exception(f'len of output({len(preds)}) != len of weights ({len(self.weights)}')

        preds_len = len(preds)
        loss = 0.0
        for pred, w in zip(preds, self.weights):
            current_loss = bdcn_loss2(pred, target, w).item()

            loss += current_loss / preds_len

        return 1.0 - loss