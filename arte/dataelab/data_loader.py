import os
import abc
from pathlib import Path

import numpy as np
from astropy.io import fits
from arte.utils.help import add_help

@add_help
class DataLoader():
    '''
    Abstract base class for data loaders
    '''
    def __init__(self):
        pass

    @abc.abstractmethod
    def assert_exists(self):
        '''Assert that the data is available'''

    @abc.abstractmethod
    def filename(self):
        '''Return the data filename, if available'''

    @abc.abstractmethod
    def load(self):
        '''Load data and return it'''


class FitsDataLoader(DataLoader):
    '''Loader for data stored into FITS files

    Parameters
    ----------

    filename: str
        FITS filename or full path
    ext: int, optional
        FITS extenstion to read. Defaults to the primary card
    transpose_axes: tuple, optional
        Axes transpose pattern as specified for np.transpose.
        Use this if time is not your first dimension
    postprocess: function, optional
        function to call after loading data and before returning it
        Must take a single parameter with the whole data array
        and return the processed data array.
    '''
    def __init__(self, filename, ext=None, transpose_axes=None, postprocess=None):
        super().__init__()
        if isinstance(filename, Path):
            self._filename = str(filename)
        else:
            self._filename = filename
        self._ext = ext
        self._transpose_axes = transpose_axes
        self._postprocess = postprocess

    def assert_exists(self):
        assert os.path.exists(self._filename)

    def filename(self):
        return self._filename

    def load(self):
        if self._ext:
            data = fits.getdata(self._filename, ext=self._ext)
        else:
            data = fits.getdata(self._filename)
        if self._transpose_axes is not None:
            print(data.shape, self._transpose_axes)
            data = data.transpose(*self._transpose_axes)
        if self._postprocess:
            data = self._postprocess(data)
        return data


class NumpyDataLoader(DataLoader):
    '''Loader for data stored into np or npz files

    Parameters
    ----------
    filename: str
        numpy filename or full path
    key: str, optional
        array key for .npz files
    transpose_axes: tuple, optional
        Axes transpose pattern as specified for np.transpose.
        Use this if time is not your first dimension
    postprocess: function, optional
        function to call after loading data and before returning it
        Must take a single parameter with the whole data array
        and return the processed data array.
    '''
    def __init__(self, filename, key=None, transpose_axes=None, postprocess=None):
        super().__init__()
        if isinstance(filename, Path):
            self._filename = str(filename)
        else:
            self._filename = filename
        if self._filename.endswith('.npz') and key is None:
            key = 'arr_0'
        self._key = key
        self._transpose_axes = transpose_axes
        self._postprocess = postprocess

    def assert_exists(self):
        assert os.path.exists(self._filename)

    def filename(self):
        return self._filename

    def load(self):
        if self._key:
            data = np.load(self._filename)[self._key]
        else:
            data = np.load(self._filename)
        if self._transpose_axes is not None:
            data = data.transpose(*self._transpose_axes)
        if self._postprocess:
            data = self._postprocess(data)
        return data

class DummyLoader(DataLoader):
    '''Dummy loader for data not stored anywhere'''
    def __init__(self):
        super().__init__()

    def assert_exists(self):
        pass

    def filename(self):
        return None

    def load(self):
        return None

class OnTheFlyLoader(DataLoader):
    '''Loader for data calculated on the fly'''
    def __init__(self, func):
        super().__init__()
        self._func = func

    def assert_exists(self):
        pass

    def filename(self):
        return None

    def load(self):
        return self._func()


class ConstantDataLoader(DataLoader):
    '''Loader for constant data'''
    def __init__(self, data):
        super().__init__()
        self._data = data

    def assert_exists(self):
        pass

    def filename(self):
        return None

    def load(self):
        return self._data

def data_loader_factory(obj, allow_none=False, name=''):
    '''
    Return the correct DataLoader instance for *obj*, which might be:
    * a string with a filename among the known ones (fits or npy)
    * a pathlib.Path instance
    * a numpy array
    * a DataLoader instance (returned unchanged)
     
    Raises ValueError if the guess fails,

    Parameters
    ----------
    obj: Loader class, numpy array, str (filename), Path instance, or None
        object to wrap in a DataLoader class
    allow_none: bool, optional
        if set to True, obj can be None. If this flag is False and obj
        is None, a ValueError exception willb be raised.
    name: str, optional
        object name for error messages.
        
    Returns
    -------
    DataLoader instance
        
    '''
    if isinstance(obj, Path):
        obj = str(obj)
    if isinstance(obj, str):
        if obj.endswith('.fits'):
            return FitsDataLoader(obj)
        elif obj.endswith('.npy') or obj.endswith('.npz'):
            return NumpyDataLoader(obj)
        else:
            raise ValueError(f'Cannot guess correct DataLoader instance for filename {obj}')
    elif isinstance(obj, np.ndarray):
        return ConstantDataLoader(obj)
    elif isinstance(obj, DataLoader):
        return obj
    elif obj is None and allow_none is True:
        return obj
    else:
        name = name or 'object'
        if allow_none:
            errstr = f'{name} must be a Loader class, a numpy array, or a filename or Path instance, or None.'
        else:
            errstr = f'{name} must be a Loader class, a numpy array, or a filename or Path instance.'
        raise ValueError(errstr)

# __oOo__
