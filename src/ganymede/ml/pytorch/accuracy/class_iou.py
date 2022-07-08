import torch


class ClassIoU_AccuracyFunctor:
    def __init__(
        self,
        classes_n,
        ignore_class_idx = -1
    ):
        self.classes_n = classes_n
        self.ignore_class_idx = -1


    def __call__(
        self,
        output : torch.Tensor,
        target : torch.Tensor
    ):
        calc_iou = 0.0
        calc_len = 0



        for idx in range(self.classes_n):
            if idx == self.ignore_class_idx: 
                continue

            output_class = output == idx
            target_class = target == idx

            if target_class.sum() == 0:
                continue

            true_positives  = (output_class == target_class) & (output_class == True)
            false_positives = (output_class == True)         & (target_class == False)
            false_negatives = (output_class == False)        & (target_class == True)

            tp = true_positives.sum()
            fp = false_positives.sum()
            fn = false_negatives.sum()

            current_iou = tp / (tp + fp + fn)

            calc_iou += current_iou
            calc_len += 1

        return calc_iou / calc_len