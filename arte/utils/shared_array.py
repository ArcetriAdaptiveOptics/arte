
import ctypes
import numpy as np
import multiprocessing as mp
import multiprocessing.sharedctypes
from functools import reduce

from arte.utils.help import add_help


@add_help
class SharedArray:
    '''
    Class for a numpy-like buffer built on top of multiprocessing.

    SharedArray instances can be passed as arguments to processes
    created with the multiprocessing module, and each of them can call the
    ndarray() method to get a local view of the array.

    No access synchronization is provided.

    As a shortcut to read/write the array, the [] operator is supported,
    thus these two statements are equivalent::

        >>> array.ndarray()[2] = 3.1415
        >>> array[2] = 3.1415

    .. warning:: SharedArray works with processes spawned with mp.Process,
                 but do not work with mp.Pool, unless an mp.Manager is used.

    .. warning:: Since 3.8 the Python standard library provides a SharedMemory
                 class that must be used instead of this.

    .. warning:: It does not work on Windows.

    Parameters
    ----------
    shape: integer sequence
         array shape
    type: numpy dtype
         array dtype
    '''

    def __init__(self, shape, dtype):

        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._ndarray = None

        n_elements = reduce(lambda x, y: x * y, self.shape)
        n_bytes = n_elements * self.dtype.itemsize
        self._shared_buf = mp.sharedctypes.RawArray(ctypes.c_byte, n_bytes)

    def __getitem__(self, key):
        return self.ndarray()[key]

    def __setitem__(self, key, value):
        self.ndarray()[key] = value

    def ndarray(self, realloc=False):
        '''
        Returns a new numpy wrapper around the buffer contents.

        Call this function after a task has been spawned the multiprocessing
        module in order to have access to the shared memory segment.

        If the array had already been accessed before passing it to the
        multiprocessing task, the task has to set `realloc` to True
        in order to reallocate a local copy of the array.

        Parameters
        ----------
        Realloc: bool, optional
            force array reallocation. Defaults to False

        Returns
        -------
        numpy.ndarray
            the shared numpy array. The array is read-write and changes
            will be immediately visible to other processes.
        '''
        if (self._ndarray is None) or realloc:
            arr = np.frombuffer(self._shared_buf, dtype=self.dtype)
            self._ndarray = arr.reshape(self.shape)
        return self._ndarray
