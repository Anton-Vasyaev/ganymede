# 3rd party
from typing import List
import torch
import numpy as np
import cv2   as cv
# project
import ganymede.draw    as g_draw
import ganymede.debug   as g_debug
import ganymede.imaging as g_image


def img_batch_to_tensor_batch(
    img_batch  : np.ndarray, 
    normalized : bool = True
) -> torch.Tensor:
    tensor = torch.from_numpy(img_batch)
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.type(torch.float32)
    
    if normalized: tensor /= 255

    return tensor


def img_list_to_tensor_batch(
    img_list   : List[np.ndarray], 
    normalized : bool = True
) -> torch.Tensor:
    img_stack_list = []
    for img in img_list:
        new_img = img.copy()
        g_image.create_channel_if_not_exist(new_img)
        new_img.shape = (1,) + new_img.shape
        img_stack_list.append(new_img)

    img_stack = np.vstack(img_stack_list)

    return img_batch_to_tensor_batch(img_stack, normalized)


def tensor_batch_to_img_batch(tensor : torch.Tensor) -> np.ndarray:
    img = tensor.permute(0, 2, 3, 1).cpu().detach().numpy()

    return img


def tensor_batch_to_img_list(tensor : torch.Tensor) -> np.ndarray:
    img_batch = tensor_batch_to_img_batch(tensor)

    new_list = []
    for img in img_batch:
        new_list.append(img)

    return new_list


def visualise_batch(
    tensor_batch            : torch.Tensor, 
    normalized_float_coords : bool = True,
    visualise_channels      : bool = False
) -> None:
    img_batch = tensor_batch.permute(0, 2, 3, 1).cpu().detach().numpy()

    if np.issubdtype(img_batch.dtype, np.floating) and normalized_float_coords:
        img_batch *= 255

    img_batch = img_batch.astype(np.uint8)

    idx = 0
    for img in img_batch:
        idx += 1

        img = img.copy()
        
        if visualise_channels and g_image.get_channels(img) != 1:
            for map_idx in range(img.shape[2]):
                map = img[...,map_idx]
                g_image.create_channel_if_not_exist(map)
                map = g_image.cast_one_channel_img(map)

                g_draw.draw_text_list(
                    map,
                    [ (f'{idx}/{len(img_batch)}, channel:{map_idx+1}/{img.shape[2]}', (0.03, 0.05), (0, 240, 0)) ],
                    0.1
                )
                g_debug.debug_img(map)
            
        else:
            g_image.create_channel_if_not_exist(img)
            if img.shape[2] == 1: img = np.repeat(img, 3, axis=2)

            g_draw.draw_text_list(
                img,
                [ (f'{idx}/{len(img_batch)}', (0.03, 0.05), (0, 240, 0)) ],
                0.1
            )
            
            g_debug.debug_img(img)

    
def make_coords_input_tensor(batch_size, height, width):
    x = ((torch.arange(width) + 0.5) / width).repeat(height, 1).view(1, height, width)

    y = ((torch.arange(height) + 0.5) / height).repeat(width, 1).view(1, width, height).transpose(1, 2)

    coords = torch.stack([x, y], 1)

    coords = coords.repeat(batch_size, 1, 1, 1)

    return coords