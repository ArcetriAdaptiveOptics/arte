import numpy as np

from arte.utils.help import add_help
from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import ConstantDataLoader
from arte.dataelab.unit_handler import UnitHandler


@add_help
class BaseData():
    '''
    Generic static data

    data_loader: instance of DataLoader or derived class
    mapper2d: function to map data into 2d. If None, data is assumed
              to be already in 2d.
    astropy_unit: if possible, astropy unit to use with the data.
    data_label: human-readable label for plot (e.g.: "Surface modal coefficients" )
    '''
    def __init__(self, data_loader, mapper2d=None, astropy_unit=None, data_label=None):
        if isinstance(data_loader, np.ndarray):
            data_loader = ConstantDataLoader(data_loader)
        try:
            # Test that the data file is there, when possible
            _ = data_loader.assert_exists()

        except AssertionError:
            NotAvailable.transformInNotAvailable(self)
            return

        self._data_loader = data_loader
        self._astropy_unit = astropy_unit
        self._data_label = data_label
        self._unit_handler = UnitHandler(wanted_unit = astropy_unit)
        if mapper2d is None:
            self._display_func = np.atleast_2d
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
        return self._unit_handler.apply_unit(self._data_loader.load())

    def get_index_of(self, *args, **kwargs):
        '''Return data slice. Override in derived classes if needed'''
        return None

    def data_label(self):
        '''Long-form data label (for plots)'''
        return self._data_label

    def data_units(self):
        '''Data unit string (for plots)'''
        return self._unit_handler.actual_unit_name()

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._unit_handler.actual_unit()

    def get_display(self):
        '''Data mapped in 2d'''
        return self._display_func(self.get_data())
    



# __oOo__