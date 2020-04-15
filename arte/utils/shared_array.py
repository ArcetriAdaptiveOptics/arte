#!/usr/bin/env python

import ctypes
import numpy as np
import multiprocessing as mp
import multiprocessing.sharedctypes
from functools import reduce

class SharedArray:
    '''
    Class for a numpy-like buffer built on top of multiprocessing.Array
    Objects of the SharedArray type can be passed between processes
    created with the multiprocessing module, and each of them can call the
    np() method to get a local view of the array.

    No access synchronization is provided
    '''
    def __init__(self):
        self.shape = None
        self.dtype = None
        self._np = None

    def alloc(self, shape, dtype):
        '''
        Allocate the shared memory buffer
        '''
        self.shape = shape
        self.dtype = dtype

        nElements = reduce( lambda x,y: x*y, self.shape)
        nBytes = nElements * self.dtype.itemsize
        self.sharedBuf = mp.sharedctypes.RawArray( ctypes.c_byte, nBytes)

    def np(self, refresh=False):
        '''
        Returns a new numpy wrapper around the buffer contents
        Call this function after a task has been spawned the multiprocessing
        module in order to have access to the shared memory segment.
        '''
        if (self._np is None) or (refresh is True):
            self._np = np.frombuffer( self.sharedBuf, dtype=self.dtype).reshape( self.shape)
        return self._np

    def refresh(self):
        '''
        Forces reallocation of the local buffer
        '''
        self.np(refresh=True)



class SharedCircularBuffer(SharedArray):
    '''
    Implements a circular buffer on top of a SharedArray.
    '''
    def __init__(self):
        SharedArray.__init__(self)
        self._circularCounter = mp.sharedctypes.RawValue( ctypes.c_uint32, 0)

    def alloc(self, nFrames, shape, dtype):

        self.nFrames = nFrames
        SharedArray.alloc( self, (nFrames,)+shape, dtype)

    def store(self, data, position=None):
        '''
        Store a record in the circular buffer. By default, the record is
        stored following an internal counter, which is then incremented.
        '''
        if position is None:
            position = self._circularCounter.value

        self.np()[ position, :] = data
        self._circularCounter.value = (position+1) % self.nFrames

    def counter(self):
        return self._circularCounter.value




