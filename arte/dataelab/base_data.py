import numpy as np

from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import ConstantDataLoader


def no_op(x):
    return x


class BaseData():
    '''
    Generic static data

    data_loader: instance of DataLoader or derived class
    mapper2d: function to map data into 2d. If None, data is assumed
              to be already in 2d.
    astropy_unit: if possible, astropy unit to use with the data.
    '''
    def __init__(self, data_loader, mapper2d=None, astropy_unit=None):
        if isinstance(data_loader, np.ndarray):
            data_loader = ConstantDataLoader(data_loader)
        else:
            data_loader = data_loader
        try:
            # Test that the data file is there, when possible
            _ = data_loader.assert_exists()

        except AssertionError:
            NotAvailable.transformInNotAvailable(self)
            return

        self._data_loader = data_loader
        self._astropy_unit = astropy_unit
        if mapper2d is None:
            self._display_func = no_op
        else:
            self._display_func = mapper2d

    def filename(self):
        '''Data filename (full path)'''
        return self._data_loader.filename()

    def get_data(self, *args, **kwargs):
        '''Raw data, selecting elements based on the indexer'''

        raw_data = self._get_not_indexed_data()
        index = self.get_index_of(*args, **kwargs)
        if index is None:
            return raw_data
        else:
            return raw_data[index]

    def _get_not_indexed_data(self):
        return self._data_loader.load()

    def get_index_of(self, *args, **kwargs):
        return None

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._astropy_unit

    def get_display(self):
        '''Data mapped in 2d'''
        return self._mapper2d(self.get_data())


# __oOo__