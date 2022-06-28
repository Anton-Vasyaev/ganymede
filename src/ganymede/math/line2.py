# python
import math


def center(line):
    (x1, y1), (x2, y2) = line

    return (x1 + x2) / 2, (y1 + y2) / 2


def left_right(line):
    (x1, _), (x2, _) = line

    min_x = min(x1, x2)
    max_x = max(x1, x2)

    return min_x, max_x


def top_bottom(line):
    (_, y1), (_, y2) = line

    min_y = min(y1, y2)
    max_y = max(y1, y2)

    return min_y, max_y


def bbox(line):
    l, r = left_right(line)
    t, b = top_bottom(line)

    return l, t, r, b


def equation(line):
    (x1, y1), (x2, y2) = line

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return a, b, c


def square_length(line):
    (x1, y1), (x2, y2) = line

    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def length(line):
    return math.sqrt(square_length(line))



def crossing(a, b):
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b

    # num - числитель
    # den - знаменталь
    x_num = (ax1 * ay2 - ay1 * ax2) * (bx1 - bx1) - (ax1 - ax2) * (bx1 * by2 - by1 * bx1)
    x_den = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx1)

    y_num = (ax1 * ay2 - ay1 * ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 * by2 - by1 * bx1)
    y_den = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx1)

    x = x_num / x_den
    y = y_num / y_den

    return x, y