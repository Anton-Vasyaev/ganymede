# python
from numbers import Number
# project
import ganymede.math.auxiliary as m_aux


def clip(
    bbox, 
    min_val, 
    max_val
):
    if isinstance(min_val, Number):
        min_w, min_h = min_val, min_val
        max_w, max_h = max_val, max_val
    else:
        min_w, min_h = min_val
        max_w, max_h = max_val

    min_w, min_h = min_val, min_val if isinstance(min_val, Number) else min_val
    max_w, max_h = max_val, max_val if isinstance(max_val, Number) else max_val

    x1, y1, x2, y2 = bbox

    x1, y1 = m_aux.clip(x1, min_w, max_w), m_aux.clip(y1, min_h, max_h)
    x2, y2 = m_aux.clip(x2, min_w, max_w), m_aux.clip(y2, min_h, max_h)

    return x1, y1, x2, y2


def scale(bbox, scale_size):
    if isinstance(scale_size, Number):
        scale_w, scale_h = scale_size, scale_size
    else:
        scale_w, scale_h = scale_size

    x1, y1, x2, y2 = bbox
    w, h           = x2 - x1, y2 - y1

    w_scale = w * scale_w / 2
    h_scale = h * scale_h / 2

    x1, y1 = x1 - w_scale, y1 - h_scale
    x2, y2 = x2 + w_scale, y2 + h_scale

    return [x1, y1, x2, y2]


def area(bbox):
    l, t, r, b = bbox

    return abs(r - l) * abs(b - t)


def center(bbox):
    l, t, r, b = bbox

    return (l + r) / 2, (t + b) / 2