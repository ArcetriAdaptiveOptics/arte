import numpy as np

from arte.time_series import TimeSeries
from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import ConstantDataLoader
from arte.dataelab.unit_handler import UnitHandler


def no_op(x):
    return x


class BaseTimeSeries(TimeSeries):
    '''
    Generic time series.

    delta_time: time interval between data samples, in seconds.
    data_loader: instance of DataLoader or derived class, or numpy array
    mapper2d: function to map data into 2d. If None, data is assumed
              to be already in 2d.
    astropy_unit: if possible, astropy unit to use with the data.
    data_label: human-readable label for plot (e.g.: "Surface modal coefficients" )
    '''
    def __init__(self, delta_time, loader_or_data, mapper2d=None, astropy_unit=None, data_label=None):
        if isinstance(loader_or_data, np.ndarray):
            loader = ConstantDataLoader(loader_or_data)
        else:
            loader = loader_or_data

        try:
            super().__init__(delta_time)

            # Also test that the data file is there, when possible
            _ = loader.assert_exists()

        except AssertionError:
            NotAvailable.transformInNotAvailable(self)
            return

        self._data_loader = loader
        self._astropy_unit = astropy_unit
        self._data_label = data_label
        self._unit_handler = UnitHandler(wanted_unit = astropy_unit)

        if mapper2d is None:
            self._display_func = np.atleast_2d
        else:
            self._display_func = mapper2d

    def _get_display_cube(self, data):
        return np.dstack([self._display_func(frame) for frame in data])

    def filename(self):
        '''Data filename (full path)'''
        return self._data_loader.filename()

    def _get_not_indexed_data(self):
        return self._unit_handler.apply_unit(self._data_loader.load())

    def get_index_of(self, *args, **kwargs):
        pass

    def data_label(self):
        return self._data_label

    def data_units(self):
        '''Data unit string (for plots)'''
        return self._unit_handler.actual_unit_name()

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._unit_handler.actual_unit()


# __oOo__
