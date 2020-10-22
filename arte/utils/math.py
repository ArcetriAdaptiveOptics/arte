import math
import numpy as np
from scipy import special
try:
    import cupy as cp
except ImportError:
    print("Can't import cupy. Use numpy.")


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def kroneckerDelta(m, n):
    if m == n:
        delta = 1
    else:
        delta = 0
    return delta


# def besselFirstKind(order, arg):
#     return special.jv(order, arg)

def besselFirstKind(order, arg, xp):
    #     import cupy as cp

    if xp == np:
        return special.jv(order, arg)
    elif xp == cp:
        return besselFirstKindOnGPU(order, arg)


def besselFirstKindOnGPU(order, arg):
    from cupyx.scipy import special
#     import cupy as cp

    if order == 0:
        return special.j0(arg)
    elif order == 1:
        return special.j1(arg)
    elif order >= 2:
        return 2 * (float(order) - 1) * besselFirstKindOnGPU
    (int(order) - 1, arg) / cp.real(arg) - besselFirstKindOnGPU(
        int(order) - 2, arg)
