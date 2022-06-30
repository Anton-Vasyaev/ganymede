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
    xc, yc         = x1 + w / 2, y1 + h / 2

    w *= scale_w 
    h *= scale_h

    x1, y1 = xc - w / 2, yc - w / 2
    x2, y2 = xc + w / 2, yc + w / 2

    return [x1, y1, x2, y2]


def area(bbox):
    l, t, r, b = bbox

    return abs(r - l) * abs(b - t)


def center(bbox):
    l, t, r, b = bbox

    return (l + r) / 2, (t + b) / 2


def iom(v, z):
    vx1, vy1, vx2, vy2 = v
    zx1, zy1, zx2, zy2 = z

    l = max(vx1, zx1)
    r = min(vx2, zx2)
    t = max(vy1, zy1)
    b = min(vy2, zy2)

    if l >= r or t >= b: return 0.0

    a_area = area(v)
    b_area = area(z)
    min_area = min(a_area, b_area)

    inter_area = area([l, t, r, b])

    return inter_area / min_area
    
