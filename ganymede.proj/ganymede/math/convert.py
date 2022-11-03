# python
import math
# project
import ganymede.core as g_core

RADIAN_PER_DEGREE = math.pi / 180
DEGREE_PER_RADIAN = 180 / math.pi


def deg2rad(degrees):
    return degrees * RADIAN_PER_DEGREE


def rad2deg(radians):
    return radians * DEGREE_PER_RADIAN


def rgb2hsv(color):
    r, g, b = color

    min_v, max_v = g_core.min_max_list([r, g, b])

    h = s = v = 0
    if   max_v == r and g >= b:
        h = 60 * (g - b) / (max_v - min_v)
    elif max_v == r and g < b:
        h = 60 * (g - b) / (max_v - min_v) + 360
    elif max_v == g:
        h = 60 * (b - r) / (max_v - min_v) + 120
    elif max_v == b:
        h = 60 * (r - g) / (max_v - min_v) + 240

    if max_v == 0:
        s = 0
    else:
        s = 1 - min_v / max_v

    v = max_v

    h = h / 360

    return h, s, v