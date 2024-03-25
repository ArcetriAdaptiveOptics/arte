# -*- coding: utf-8 -*-
#########################################################
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-15  Created
#
#########################################################

import numpy as np

def circular_mask(size, center=None, diameter=1.0, obstruction=0.0):
    '''
    Creates a circular mask for *size*x*size* elements, where elements
    whose distance from *center* is higher than *ob/2* and smaller
    than *diameter/2* are set to one, and all others at zero.
    Both *diameter* and *obstruction* are in percentage (0..1),
    where 1 is the full aperture.
    *center* should be a sequence of two coordinates. If omitted or None,
    it will be set to [*size/2*,*size/2*].
    
    See also arte.types.mask for the class CircularMask and derived classes
    '''
    if center is None:
        center = [(size-1)/2, (size-1)/2]

    xx, yy = np.ogrid[:size,:size]
    dist = np.sqrt((xx - center[0])**2 + (yy-center[1])**2)

    mask = dist <= diameter*size/2
    mask *= dist >= obstruction*size/2
    return mask

# ___oOo___
