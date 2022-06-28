def add(a, b):
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 + bv1, av2 + bv2, av3 + bv3


def add_val(t, val):
    v1, v2, v3 = t

    return v1 + val, v2 + val, v3 + val 


def sub(a, b):
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 - bv1, av2 - bv2, av3 - bv3


def sub_val(t, val):
    v1, v2, v3 = t

    return v1 - val, v2 - val, v3 - val 


def mul(a, b):
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 * bv1, av2 * bv2, av3 * bv3


def mul_val(t, val):
    v1, v2, v3 = t

    return v1 * val, v2 * val, v3 * val 


def div(a, b):
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 / bv1, av2 / bv2, av3 / bv3


def div_val(t, val):
    v1, v2, v3 = t

    return v1 / val, v2 / val, v3 / val 


def minus(t):
    v1, v2, v3 = t

    return -v1, -v2, -v3


def sum(values):
    sum_t = 0.0, 0.0, 0.0
    for v in values:
        sum_t = add(sum_t, v)

    return sum_t


def mean(values):
    sum_t = sum(values)

    return div_val(sum_t, len(values))