# -*- coding: utf-8 -*-

import numpy as np
from arte.utils.help import ThisClassCanHelp, add_to_help
from arte.utils.shared_array import SharedArray

class GenericCircularBuffer(ThisClassCanHelp):
    '''
    Implements a generic circular buffer using any class with a 
    numpy-like interface (must define a constructor with shape and dtype
    parameters, and the constructed object must have a read/write [] operator)
    '''
    def __init__(self, n_frames, shape, dtype, constructor):
        self._buf = constructor((n_frames,)+shape, dtype=dtype)
        self._len = n_frames
        self._counter = constructor((1,), dtype=np.uint32)
        self._position = constructor((1,), dtype=np.uint32)

    @add_to_help
    def store(self, data, position=None):
        '''
        Store a record in the circular buffer. By default, the record is
        stored following an internal counter, which is then incremented.
        '''
        if position is None:
            position = self._position[0]

        self._buf[position,:] = data
        self._counter[0] = self._counter[0] + 1
        self._position[0] = self._counter[0] % self._len

    @add_to_help
    def get(self, position):
        '''Returns a frame from the circular buffer'''
        return self._buf[position,:]

    @add_to_help
    def position(self):
        '''Returns the current position in the circular buffer'''
        return self._position[0]

    @add_to_help
    def counter(self):
        '''Returns the total number of stored frames'''
        return self._counter[0]  

class NumpyCircularBuffer(GenericCircularBuffer):
    '''
    Implements a circular buffer on top of a numpy array
    '''
    def __init__(self, n_frames, shape, dtype):
        super().__init__(n_frames, shape, dtype, np.zeros)

class SharedCircularBuffer(GenericCircularBuffer):
    '''
    Implements a circular buffer on top of a SharedArray
    '''
    def __init__(self, n_frames, shape, dtype):
        super().__init__(n_frames, shape, dtype, SharedArray)