# python
from typing import Tuple, Collection, cast
# project
import math.auxiliary as m_aux
from .primitives import AlgTuple3


def add(a : AlgTuple3, b : AlgTuple3) -> AlgTuple3:
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 + bv1, av2 + bv2, av3 + bv3


def add_val(t : AlgTuple3, val : float) -> AlgTuple3:
    v1, v2, v3 = t

    return v1 + val, v2 + val, v3 + val 


def sub(a : AlgTuple3, b : AlgTuple3) -> AlgTuple3:
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 - bv1, av2 - bv2, av3 - bv3


def sub_val(t : AlgTuple3, val : float) -> AlgTuple3:
    v1, v2, v3 = t

    return v1 - val, v2 - val, v3 - val 


def mul(a : AlgTuple3, b : AlgTuple3) -> AlgTuple3:
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 * bv1, av2 * bv2, av3 * bv3


def mul_val(t : AlgTuple3, val : float) -> AlgTuple3:
    v1, v2, v3 = t

    return v1 * val, v2 * val, v3 * val 


def div(a : AlgTuple3, b : AlgTuple3) -> AlgTuple3:
    av1, av2, av3 = a
    bv1, bv2, bv3 = b

    return av1 / bv1, av2 / bv2, av3 / bv3


def div_val(t : AlgTuple3, val : float) -> AlgTuple3:
    v1, v2, v3 = t

    return v1 / val, v2 / val, v3 / val 


def minus(t : AlgTuple3) -> AlgTuple3:
    v1, v2, v3 = t

    return -v1, -v2, -v3


def sum(values : Collection[AlgTuple3]) -> AlgTuple3:
    sum_t = 0.0, 0.0, 0.0
    for v in values:
        sum_t = add(sum_t, v)

    return sum_t


def mean(values : Collection[AlgTuple3]) -> AlgTuple3:
    sum_t = sum(values)

    return div_val(sum_t, len(values))


def clip(
    value      : AlgTuple3, 
    dimensions : Tuple[AlgTuple3, AlgTuple3, AlgTuple3]
) -> AlgTuple3:
    v1_dim, v2_dim, v3_dim = dimensions

    v1_min, v1_max = v1_dim
    v2_min, v2_max = v2_dim
    v3_min, v3_max = v3_dim

    v1, v2, v3 = value

    v1 = cast(float, m_aux.clip(v1, v1_min, v1_max))
    v2 = cast(float, m_aux.clip(v2, v2_min, v2_max))
    v3 = cast(float, m_aux.clip(v3, v3_min, v3_max))

    return v1, v2, v3