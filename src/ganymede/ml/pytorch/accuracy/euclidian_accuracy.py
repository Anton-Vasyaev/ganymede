# 3rd party
import torch


def euclidian_accuracy(output, target):
    dim_dist = output - target

    pows = dim_dist ** 2

    division = pows[:, :, 0] + pows[:, :, 1]

    distance = torch.sqrt(division)

    mean = float(distance.mean())

    return 1.0 - mean