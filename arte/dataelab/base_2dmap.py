from functools import cached_property
import numpy as np
from arte.dataelab.base_data import BaseData

class Base2dMap(BaseData):
    '''
    2d map. Initialize with a 2d map
    where valid elements are 1 and not used elements are 0.
    '''
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
        
        Result
        ------
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
