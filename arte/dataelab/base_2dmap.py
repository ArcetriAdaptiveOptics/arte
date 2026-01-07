from functools import cached_property
import numpy as np
from arte.dataelab.base_data import BaseData

class Base2dMap(BaseData):
    """Two-dimensional map for spatial data representation.
    
    This class represents 2D spatial maps derived from time-series data,
    such as camera images, wavefront maps on telescope pupil, or deformable
    mirror shapes. It handles mapping between 1D data vectors and 2D spatial
    representations.
    
    The map defines which elements are valid (value=1) and which are not used
    (value=0), enabling proper reshaping of vectorized data into 2D images.
    
    Parameters
    ----------
    map2d : ndarray
        2D boolean or integer array where valid elements are 1 (or True)
        and unused elements are 0 (or False)
    
    Examples
    --------
    >>> # Create a circular pupil map
    >>> pupil = np.zeros((100, 100))
    >>> y, x = np.ogrid[-50:50, -50:50]
    >>> mask = x**2 + y**2 <= 40**2
    >>> pupil[mask] = 1
    >>> map2d = Base2dMap(pupil)
    >>> print(map2d.nvalid)  # Number of valid pixels
    
    >>> # Remap 1D vector data to 2D image
    >>> data_1d = np.random.randn(100, map2d.nvalid)  # 100 frames
    >>> data_2d = map2d.remap_image(data_1d)  # Shape: (100, 100, 100)
    """
    def __init__(self, map2d):
        super().__init__(map2d)

    @staticmethod
    def from_idx1d(shape, idx1d):
        f2d = np.zeros(shape, dtype=bool)
        f2d.flat[idx1d] = True
        return Base2dMap(f2d)

    @cached_property
    def nvalid(self):
        '''Number of valid elements'''
        nonzero = np.nonzero(self.get_data())
        return len(nonzero[0])

    def as_mask(self):
        ''''
        Return a mask suitable to be used in a masked array
        '''
        return (1 - self.get_data()).astype(bool)

    def remap_image(self, data):
        '''
        Remap a 2d time series data into a 3d cube, where each
        cube slice is a 2d image. If a 1d array is passed,
        the resulting 3d array will have a first dimension with length 1.
        
        Parameters
        ----------
        data: ndarray
            2d numpy array [time, values] or 1d numpy array [values]
        
        Returns
        -------
        frame: ndarray
            3d numpy array [time, x, y]
        '''
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)

        nonzero = np.nonzero(self.get_data())
        frame = np.zeros((len(data), self.shape[0], self.shape[1]), dtype=data.dtype)
        if len(nonzero[0]) != data.shape[1]:
            raise ValueError(f'Cannot build display: data shape and subap map shape do not match: %s and %d elements' % (data.shape, len(nonzero[0])))
        frame[:, nonzero[0], nonzero[1]] = data
        return frame
