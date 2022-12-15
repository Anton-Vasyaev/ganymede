# 3rd party
import torch
import numpy as np


def make_img_as_tensor(numpy_img, device=torch.device('cpu')):
    img_t = torch.from_numpy(numpy_img).to(device).float()
    img_t = img_t.permute(2, 0, 1).type(torch.float32)
    img_t = img_t.view((1,) + img_t.shape)
    img_t /= 255.0

    return img_t


def cast_tensor_rgb_to_gray(tensor):
    device = tensor.device

    new_tensor = torch.zeros(
        (tensor.shape[0],) + (1,) + tensor.shape[2:4],
        dtype=torch.float32,
        device=device
    )
    new_tensor[:,0,:,:] = (tensor[:,0,:,:] + tensor[:,1,:,:] + tensor[:,2,:,:]) / 3.0

    return new_tensor


def select_or_make_tensor(img, device=torch.device('cpu')):
    if type(img) == torch.Tensor:
        return img
    else:
        return make_img_as_tensor(
            img,
            device
        )