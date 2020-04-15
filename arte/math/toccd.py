# -*- coding: utf-8 -*-
##########################################
# who       when        what
# --------  ----------  ------------------
# apuglisi  2019-09-28  Created
#
##########################################


from arte.math.factors import lcm
from arte.utils.rebin import rebin


def toccd(a, newshape, set_total=None):
    '''
    Clone of oaalib's toccd() function, using least common multiple
    to rebin an array similar to opencv's INTER_AREA interpolation.
    '''
    if a.shape == newshape:
        return a

    if len(a.shape) != 2:
        raise ValueError('Input array shape is %s instead of 2d, cannot continue:' % str(a.shape))

    if len(newshape) != 2:
        raise ValueError('Output shape is %s instead of 2d, cannot continue' % str(newshape))

    if set_total is None:
        set_total = a.sum()

    mcmx = lcm(a.shape[0], newshape[0])
    mcmy = lcm(a.shape[1], newshape[1])

    temp = rebin(a, (mcmx, a.shape[1]), sample=True)
    temp = rebin(temp, (newshape[0], a.shape[1]))
    temp = rebin(temp, (newshape[0], mcmy), sample=True)
    rebinned = rebin(temp, newshape)

    return rebinned / rebinned.sum() * set_total

