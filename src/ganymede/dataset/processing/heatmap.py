# 3rd party
import numpy as np
import torch


_gaussians = {}

def generate_keypoints_heatmap(size, keypoints, sigma=5):
    w, h = size

    heatmap = torch.zeros(1, len(keypoints), h, w)

    tmp_size = sigma * 3

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians \
                else torch.tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g

    for k_idx in range(len(keypoints)):
        x, y = keypoints[k_idx]

        # Heatmap pixel per output pixel
        mg_x = int(x * w)
        mg_y = int(y * h)

        x1, y1 = int(mg_x - tmp_size), int(mg_y - tmp_size)

        x2, y2 = int(mg_x + tmp_size + 1), int(mg_y + tmp_size + 1)
        if x1 >= w or y1 >= h or x2 < 0 or y2 < 0: continue

        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, w)
        img_y_min, img_y_max = max(0, y1), min(y2, h)

        heatmap[0, k_idx, img_y_min:img_y_max, img_x_min:img_x_max] = \
            g[g_y_min:g_y_max, g_x_min:g_x_max]

    return heatmap