import math
from scipy import special


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def kroneckerDelta(m, n):
    if m == n:
        delta = 1
    else:
        delta = 0
    return delta


def besselFirstKind(order, arg):
    return special.jv(order, arg)
