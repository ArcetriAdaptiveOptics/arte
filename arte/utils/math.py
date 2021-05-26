import math


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def kroneckerDelta(m, n):
    if m == n:
        delta = 1
    else:
        delta = 0
    return delta


def besselFirstKind(order, arg):
    from scipy import special
    return special.jv(order, arg)


def besselFirstKindOnGPU(order, arg):
    import cupy as cp
    from cupyx.scipy import special
    if order == 0:
        bess = special.j0(arg)
    elif order == 1:
        bess = special.j1(arg)
    elif order >= 2:
        bess = 2 * (float(order) - 1) * besselFirstKindOnGPU(int(order) - 1, arg) / cp.real(arg) - besselFirstKindOnGPU(
            int(order) - 2, arg)
        if not cp.all(arg):
            zero_idx = cp.where(arg == 0)
            bess[zero_idx] = 0
    return bess
