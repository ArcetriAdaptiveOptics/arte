from functools import cached_property

import numpy as np

class SubapMap:
    '''
    2d subaperture map. Initialize by calling *set_map* with a 2d map
    where valid subapertures are 1 and not used subapertures are 0.
    '''
    def __init__(self, map2d):
        # override in derived classes
        self._subap_map = map2d

    @cached_property
    def nsubaps(self):
        '''Number of active subapertures'''
        if self._subap_map is None:
            return 0
        else:
            nonzero = np.nonzero(self._subap_map)
            return len(nonzero[0])

    @cached_property
    def shape(self):
        return self._subap_map.shape

    def as_subap_image(self, data):
        '''
        Remap the 2d time series data into a 3d cube, where each
        cube slice is a 2d pupil image.
        
        Parameters
        ----------
        data: ndarray
            2d numpy array [time, values]
        
        Result
        ------
        frame: ndarray
            3d numpy array [time, x, y]
        '''
        if self._subap_map is not None:
            nonzero = np.nonzero(self._subap_map)
            ss = self._subap_map.shape
            frame = np.zeros((len(data), ss[0], ss[1]), dtype=np.float32)
            if len(nonzero[0]) != data.shape[1]:
                raise ValueError('Cannot build display: data shape and subap map shape do not match')
            frame[:, nonzero[0], nonzero[1]] = data
            return frame
        else:
            raise Exception('Cannot build display: subaperture map was not set')
