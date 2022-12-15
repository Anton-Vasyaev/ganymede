# 3rd party
import torch
import numpy as np
# project
import ganymede.opencv as g_cv


SUBSTRACT_CHANNELS_VALUES = [ 103.939, 116.779, 123.68]

def dexined_rgb_input_processor(input_t):
    view = input_t.permute(0, 2, 3, 1)
    view -= torch.tensor(SUBSTRACT_CHANNELS_VALUES)


def dexined_gray_input_processor(input_t):
    input_t -= np.array(SUBSTRACT_CHANNELS_VALUES).mean()


def dexined_input_processor(input_t):
    channels = int(input_t.shape[1])

    if channels == 1:
        dexined_gray_input_processor(input_t)
    elif channels == 3:
        dexined_rgb_input_processor(input_t)
    else:
        raise Exception(f'Unknown numbers of channels for dexined input:{channels}.')


DEXINED_INPUT_PROCESSORS = {
    1 : dexined_gray_input_processor,
    3 : dexined_rgb_input_processor
}


def debug_dexined_input(input_t):
    input_t = input_t.clone()
    input_t   = input_t.permute(0, 2, 3, 1)
    img_batch = input_t.cpu().detach().numpy()

    img_c = img_batch.shape[3]
    
    if img_c == 3:
        for c_idx in range(3):
            img_batch[:, :, :, c_idx] += SUBSTRACT_CHANNELS_VALUES[c_idx]
    elif img_c == 1:
        img_batch += np.array(SUBSTRACT_CHANNELS_VALUES).mean()

    img_batch /= 255

    for img in img_batch:
        g_cv.imshow('debug', img)