import torch

def heatmap_keypoint_detection(out, sigmoid = True):
    if sigmoid: out = torch.sigmoid(out)

    o_b, o_k, o_h, o_w = out.shape

    out = out.view(o_b, o_k, o_h * o_w)

    argmax_o = out.argmax(axis=2)
    x_coords = argmax_o %  o_w
    y_coords = torch.div(argmax_o, o_w, rounding_mode='trunc') # is //

    x_coords = x_coords.view(x_coords.shape + (1,)) / o_w
    y_coords = y_coords.view(y_coords.shape + (1,)) / o_h

    coords = torch.concat([x_coords, y_coords], axis=2)

    return coords

