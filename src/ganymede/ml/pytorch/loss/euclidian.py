import torch


def euclidian_loss(output, target):
    dim_dist = output - target

    pows = dim_dist ** 2

    division = pows[:, :, 0] + pows[:, :, 1]

    distance = torch.sqrt(division)

    return distance.mean()