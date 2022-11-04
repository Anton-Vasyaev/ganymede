# python
from typing import Collection, Tuple, cast
# project
import ganymede.math.auxiliary as m_aux
from .primitives import AlgTuple2


def add(a : AlgTuple2, b : AlgTuple2) -> AlgTuple2:
    av1, av2 = a
    bv1, bv2 = b

    return av1 + bv1, av2 + bv2


def add_val(t : AlgTuple2, val : float) -> AlgTuple2:
    v1, v2 = t

    return v1 + val, v2 + val


def sub(a : AlgTuple2, b : AlgTuple2) -> AlgTuple2:
    av1, av2 = a
    bv1, bv2 = b

    return av1 - bv1, av2 - bv2


def sub_val(t : AlgTuple2, val : AlgTuple2) -> AlgTuple2:
    v1, v2 = t

    return v1 - val, v2 - val


def mul(a : AlgTuple2, b : AlgTuple2) -> AlgTuple2:
    av1, av2 = a
    bv1, bv2 = b

    return av1 * bv1, av2 * bv2


def mul_val(t : AlgTuple2, val : float) -> AlgTuple2:
    v1, v2 = t

    return v1 * val, v2 * val


def div(a : AlgTuple2, b : AlgTuple2) -> AlgTuple2:
    av1, av2 = a
    bv1, bv2 = b

    return av1 / bv1, av2 / bv2


def div_val(t : AlgTuple2, val : float) -> AlgTuple2:
    v1, v2 = t

    return v1 / val, v2 / val


def minus(t : AlgTuple2) -> AlgTuple2:
    v1, v2 = t

    return -v1, -v2


def sum(values : Collection[AlgTuple2]) -> AlgTuple2:
    sum_t = 0.0, 0.0
    for v in values:
        sum_t = add(sum_t, v)

    return sum_t


def mean(values : Collection[AlgTuple2]) -> AlgTuple2:
    sum_t = sum(values)

    return div_val(sum_t, len(values))


def clip(value : AlgTuple2, dimensions : Tuple[AlgTuple2, AlgTuple2]) -> AlgTuple2:
    v1_dim, v2_dim = dimensions

    v1_min, v1_max = v1_dim
    v2_min, v2_max = v2_dim

    v1, v2 = value

    v1 = cast(float, m_aux.clip(v1, v1_min, v1_max))
    v2 = cast(float, m_aux.clip(v2, v2_min, v2_max))

    return v1, v2