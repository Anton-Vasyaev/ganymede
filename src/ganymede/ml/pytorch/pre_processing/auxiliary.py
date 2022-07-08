import torch


def class_map_to_prob_map(
    class_map : torch.Tensor,
    class_n   : int
):
    if len(class_map.shape) != 3:
        b, c, h, w = class_map.shape
        if c != 1:
            raise ValueError(f'invalid channels in 1 axis in class map:{c}')
        
        class_map = class_map.view(b, h, w)

    prob_map = torch.zeros((b, class_n, h, w), dtype=torch.float32)

    for class_idx in range(class_n):
        prob_map[:,class_idx,...][class_map == class_idx] = 1.0

    return prob_map