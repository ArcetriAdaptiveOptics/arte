from arte.time_series import TimeSeries
from arte.utils.not_available import NotAvailable

def no_op(x):
    return x
class BaseTimeseries(TimeSeries):
    '''
    Generic time series.

    delta_time: time interval between data samples, in seconds.
    data_loader: instance of DataLoader or derived class
    mapper2d: function to map data into 2d. If None, data is assumed
              to be already in 2d.
    astropy_unit: if possible, astropy unit to use with the data.
    '''
    def __init__(self, delta_time, data_loader, mapper2d=None, astropy_unit=None):
        try:
            TimeSeries.__init__(delta_time)

            # Also test that the data file is there, when possible
            _ = data_loader.assert_exist()

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

    def _get_not_indexed_data(self):
        return self._data_loader.load()

    def get_index_of(self, *args, **kwargs):
        pass

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._astropy_unit

# __oOo__