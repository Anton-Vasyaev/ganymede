# python
from numbers import Number
# project
import ganymede.math.auxiliary as m_aux


def clip(
    rectangle, 
    min_val, 
    max_val
):
    if isinstance(min_val, Number):
        min_w, min_h = min_val, min_val
    else:
        min_w, min_h = min_val

    min_w, min_h = min_val, min_val if isinstance(min_val, Number) else min_val
    max_w, max_h = max_val, max_val if isinstance(max_val, Number) else max_val

    x1, y1, w, h = rectangle
    x2, y2 = x1 + w, y1 + h

    x1, y1 = m_aux.clip(x1, min_w, max_w), m_aux.clip(y1, min_h, max_h)
    x2, y2 = m_aux.clip(x2, min_w, max_w), m_aux.clip(y2, min_h, max_h)

    w, h = x2 - x1, y2 - y1

    return x1, y1, w, h