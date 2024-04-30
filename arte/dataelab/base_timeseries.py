import numpy as np

from arte.time_series import TimeSeries
from arte.utils.help import modify_help
from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import ConstantDataLoader
from arte.dataelab.unit_handler import UnitHandler
from astropy import units as u


class BaseTimeSeries(TimeSeries):
    '''
    Generic time series.

    delta_time: time interval between data samples, in seconds.
    data_loader: instance of DataLoader or derived class, or numpy array
    time_vector: instance of DataLoader or derived class, or numpy_array
    mapper2d: function to map data into 2d. If None, data is assumed
              to be already in 2d.
    astropy_unit: if possible, astropy unit to use with the data.
    data_label: human-readable label for plot (e.g.: "Surface modal coefficients" )
    '''
    def __init__(self, loader_or_data, time_vector=None, mapper2d=None, astropy_unit=None, data_label=None):

        if isinstance(loader_or_data, np.ndarray):
            data_loader = ConstantDataLoader(loader_or_data)
        else:
            data_loader = loader_or_data

        if isinstance(time_vector, np.ndarray):
            time_vector_loader = ConstantDataLoader(time_vector)
        else:
            time_vector_loader = time_vector

        try:
            super().__init__()

            # Also test that the data file is there, when possible
            _ = data_loader.assert_exists()
            if time_vector_loader is not None:
                _ = time_vector_loader.assert_exists()

        except AssertionError:
            NotAvailable.transformInNotAvailable(self)
            return

        self._data_loader = data_loader
        self._time_vector_loader = time_vector_loader
        self._astropy_unit = astropy_unit
        self._data_label = data_label
        self._unit_handler = UnitHandler(wanted_unit = astropy_unit)

        if mapper2d is None:
            self._display_func = self._no_op_display
        else:
            self._display_func = mapper2d

    def _no_op_display(self, x):
        return np.atleast_3d(x)

    def filename(self):
        '''Data filename (full path)'''
        return self._data_loader.filename()

    def _get_not_indexed_data(self):
        '''Reimplementation for lazy loading and astropy units'''
        return self._unit_handler.apply_unit(self._data_loader.load())

    def _get_time_vector(self):
        '''Reimplementation for lazy loading'''
        if self._time_vector_loader is not None:
            return self._time_vector_loader.load()
        else:
            return super()._get_time_vector()

    def get_index_of(self, *args, **kwargs):
        '''Should be overridden in derived classes'''
        pass

    def data_label(self):
        return self._data_label

    def data_units(self):
        '''Data unit string (for plots)'''
        return self._unit_handler.actual_unit_name()

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._unit_handler.actual_unit()

    # Override to provide custom displays
    def _get_display_cube(self, data_to_display):
        '''Generate a 3d cube for display'''
        return self._display_func(data_to_display)

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def get_display(self, *args, times=None, **kwargs):
        '''Display cube for the specified time interval'''
        data_to_display = self.get_data(*args, times=times, **kwargs)
        display_data = self._get_display_cube(data_to_display)
        if isinstance(display_data, u.Quantity):
            return display_data.value
        else:
            return display_data

# __oOo__
