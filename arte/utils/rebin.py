#!/usr/bin/env python
'''

# who       when        what
# -------  ----------  ---------------------------------
# apuglisi  2022-02-30  Fixed bug in detecting downsampling
#                       on just one axis
# apuglisi  2020-04-15  Added sample option, added exceptions,
#                       allow any kind of sequence for new_shape
#           2012-11-20  Created from git://gist.github.com/1348792.git
'''
import numpy as np


def rebin(a, new_shape, sample=False):
    """
    Replacement of IDL's rebin() function for 2d arrays.

    Resizes a 2d array by averaging or repeating elements.
    New dimensions must be integral factors of original dimensions,
    otherwise a ValueError exception will be raised.

    Parameters
    ----------
    a : ndarray
        Input array.
    new_shape : 2-elements sequence
        Shape of the output array
    sample : bool
        if True, when reducing the array side elements are set
        using a nearest-neighbor algorithm instead of averaging.
        This parameter has no effect when enlarging the array.

    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array  the data are averaged,
        unless the sample parameter is set.
        If the new shape is bigger array elements are repeated.

    Raises
    ------
    ValueError
        in the following cases:
         - new_shape is not a sequence of 2 values that can be converted to int
         - new dimensions are not an integral factor of original dimensions
    NotImplementedError
         - one dimension requires an upsampling while the other requires
           a downsampling

    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
    >>> rebin(b, (2, 3)) #downsize
    array([[0. , 0.5, 1. ],
           [2. , 2.5, 3. ]])
    >>> rebin(b, (2, 3), sample=True) #downsize
    array([[0, 0, 1],
           [2, 2, 3]])
    """

    # unpack early to allow any 2-length type for new_shape
    m, n = map(int, new_shape)

    if a.shape == (m, n):
        return a

    M, N = a.shape

    if m <= M and n <= N:
        if (M//m != M/m) or (N//n != N/n):
            raise ValueError('Cannot downsample by non-integer factors')

    elif M <= m and M <= m:
        if (m//M != m/M) or (n//N != n/N):
            raise ValueError('Cannot upsample by non-integer factors')

    else:
        raise NotImplementedError('Up- and down-sampling in different axes '
                                  'is not supported')

    if sample:
        slices = [slice(0, old, float(old) / new)
                  for old, new in zip(a.shape, (m, n))]
        idx = np.mgrid[slices].astype(np.int32)
        return a[tuple(idx)]
    else:
        if m <= M and n <= N:
            return a.reshape((m, M//m, n, N//n)).mean(3).mean(1)
        elif m >= M and n >= M:
            return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

# __oOo__
