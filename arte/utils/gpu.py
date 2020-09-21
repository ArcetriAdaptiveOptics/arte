#!/usr/bin/env python3
#
#########################################################
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2020-07-24  Created
#
#########################################################
'''
Utilites for CPU/GPU agnostic code using CUPY.

All functions in this module are safe to call even when cupy is not installed,
because they will use cupy only when an user-supplied value is a cupy
array (and thus cupy has already been imported).
'''

from functools import wraps


def is_numpy_or_cupy_array(arr):
    '''Returns True if the argument is either a numpy or a cupy array

    The check is performed looking at object attributes, therefore
    can be called without importing the cupy module.
    '''
    return hasattr(arr, '__array_function__') and \
           hasattr(arr, '__array_ufunc__')
    

def is_cupy_array(arr):
    '''Returns True if the argument is a cupy array.

    The check is performed looking at object attributes, therefore
    can be called without importing the cupy module.
    '''
    return hasattr(arr, '__array_function__') and \
           hasattr(arr, '__array_ufunc__') and \
           hasattr(arr, 'device')


def is_numpy_array(arr):
    '''Returns True if the argument is a numpy array.

    The check is performed looking at object attributes, therefore
    can be called without importing the cupy module.
    '''
    return hasattr(arr, '__array_function__') and \
           hasattr(arr, '__array_ufunc__') and \
           not hasattr(arr, 'device')


def get_array_module(arr):
    '''
    A reimplementation of cupy.get_array_module() that works
    also when cupy is not installed.
    '''
    if is_cupy_array(arr):
        import cupy
        return cupy
    else:
        import numpy
        return numpy


def _format_owner(owner):

    if owner is None:
        return 'None'

    owner = owner.__class__.__name__
    if hasattr(owner, 'name'):
       owner += '(%s)' % owner.name
    return owner


def from_GPU(value, owner=None, quiet=False):
    '''
    If *value* is a cupy array, transfer it to the host
    and return the corresponding numpy array.

    All other values are left alone. It is safe to call
    this function with any kind of object, and also when
    cupy is not installed. In the latter case, of course
    no transfer will be done.
    '''
    if is_cupy_array(value):

        if not quiet:
            import cupy
            infostr = 'from_GPU: transferring %s %s' % (value.shape, value.dtype)
            infostr += ' from device %d' % cupy.cuda.runtime.getDevice()
            infostr += ' (owner %s)' % _format_owner(owner)
            print(infostr)
        return value.get()
    else:
        return value


def to_GPU(array, owner=None, quiet=False):
    '''
    Transfer *array* to the GPU and return the corresponding cupy array.
    '''
    if array is None:
        return array

    if not is_cupy_array(array):
        import cupy

        if not quiet:
            if is_numpy_array(array):
                infostr = 'to_GPU: transferring %s %s' % (array.shape, array.dtype)
            else:
                infostr = 'to_GPU: transferring object %s' % str(array)
            infostr += ' to device %d' % cupy.cuda.runtime.getDevice()
            infostr += ' (owner %s)' % _format_owner(owner)
            print(infostr)

        array = cupy.array(array)
    return array


def _make_sure_first_arg_is_little_endian(f):
    '''
    Decorator to enforce little endianess with cupy.

    This is a workaround for the cupy issue with FITS files:
    https://github.com/cupy/cupy/issues/3652

    It will analyze the first argument and, if it is
    a big-endian array (like the ones returned by astropy.io.fits.getdata),
    convert it to a little-endian one before passing it to the function.

    Can be used to "patch" the cupy routines as follows::

        cupy.array = _make_sure_first_arg_is_little_endian(cupy.array)
        cupy.asarray = _make_sure_first_arg_is_little_endian(cupy.asarray)
    '''
    @wraps(f)
    def wrapper(data, *args, **kwargs):
        import sys
        import numpy

        if isinstance(data, numpy.ndarray):
            # Use names compatible with sys.byteorder
            endianess_map = {
                '>': 'big',
                '<': 'little',
                '=': sys.byteorder,
                '|': 'not relevant'
            }
            endianess = endianess_map[data.dtype.byteorder]
            if endianess == 'big':
               print('Array %s %s must be converted from big to little endian' %
                     (str(data.dtype), str(data.shape)))
               data = data.astype(data.dtype.newbyteorder('<'))

        return f(data, *args, **kwargs)

    return wrapper


def cupy_patch(cupy):
    '''
    Patch the passed cupy module with our workarounds for FITS files.
    '''
    cupy.array = _make_sure_first_arg_is_little_endian(cupy.array)
    cupy.asarray = _make_sure_first_arg_is_little_endian(cupy.asarray)    


import threading
import queue

class WorkerThread:
    '''
    Generic worker thread, using queues to communicate.

    After being initialized with *func*, calling *trigger()*
    will queue values to be processed by *func*, while
    *result()* will fetch the next result from *func*.

    *result()* may block until the next result is available.

    The thread will exit once all references to it go out
    of scope. Call *terminate()* to exit the thread explicitly.
    '''

    sentinel = object()

    def __init__(self, func):
        self._qin = queue.Queue()
        self._qout = queue.Queue()
        self._func = func
        self.thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.thread.start()

    def __del__(self):
        self.terminate()

    def worker_loop(self):
        while True:
            value = self._qin.get()
            if value is self.sentinel:
                return
            result = self._func(value)
            self._qout.put(result)

    def trigger(self, value):
        self._qin.put(value)

    def result(self):
        return self._qout.get()

    def terminate(self):
        self._qin.put(self.sentinel)


# ___oOo___
