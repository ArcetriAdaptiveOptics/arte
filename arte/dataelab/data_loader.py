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
    '''Loader for data stored into FITS files'''
    def __init__(self, filename, ext=None):
        super().__init__()
        if isinstance(filename, Path):
            self._filename = str(filename)
        else:
            self._filename = filename
        self._ext = ext

    def assert_exists(self):
        assert os.path.exists(self._filename)

    def filename(self):
        return self._filename

    def load(self):
        if self._ext:
            return fits.getdata(self._filename, ext=self._ext)
        else:
            return fits.getdata(self._filename)


class NumpyDataLoader(DataLoader):
    '''Loader for data stored into np or npz files'''
    def __init__(self, filename, key=None):
        super().__init__()
        if isinstance(filename, Path):
            self._filename = str(filename)
        else:
            self._filename = filename
        if self._filename.endswith('.npz') and key is None:
            key = 'arr_0'
        self._key = key

    def assert_exists(self):
        assert os.path.exists(self._filename)

    def filename(self):
        return self._filename

    def load(self):
        if self._key:
            return np.load(self._filename)[self._key]
        else:
            return np.load(self._filename)


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


class ConstantDataLoader(OnTheFlyLoader):
    '''Loader for constant data'''
    def __init__(self, data):
        super().__init__(lambda: data)

# __oOo__
