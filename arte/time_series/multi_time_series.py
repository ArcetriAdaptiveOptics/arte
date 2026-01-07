
import numpy as np
import astropy.units as u

from arte.utils.help import modify_help
from arte.time_series.time_series import TimeSeries

class MultiTimeSeries(TimeSeries):
    '''
    Join multiple TimeSeries objects with incompatible time sampling.
    
    After being initialized with a list of
    :class:`~arte.time_series.time_series.TimeSeries` objects, which can
    have different ensemble sizes and length, it behaves as if all the data
    was part of a single ensemble. Operations that make calculations
    time-wise (like 
    :meth:`~arte.time_series.time_series.TimeSeries.time_std()`) work in any
    case. The ensemble-wise ones
    (like :meth:`~arte.time_series.time_series.TimeSeries.ensemble_std()`)
    only work if deltaTime is the same
    across all series, and raise an Exception otherwise.
    '''
    def __init__(self, *args):
        super().__init__()

        self._series = []
        for v in args:
            self.add_series(v)

    def add_series(self, series):
        '''
        Adds a new series to this MultiTimeSeries instance
        
        Parameters
        ----------
        series: :class:`~arte.time_series.time_series.TimeSeries` or :class:`~arte.time_series.time_series.TimeSeriesWithInterpolation` instance
           the series to be added        
        '''
        self._series.append(series)

    def _get_not_indexed_data(self):
        return np.hstack([v._get_not_indexed_data()
                          for v in self._series])
        
    def is_homogeneous(self, *args, **kwargs):
        '''
        Check if selected series have compatible sampling rates.
        
        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`get_index_of` for series selection.
        
        Returns
        -------
        bool
            True if all selected series have identical `delta_time`,
            False otherwise.
        
        Notes
        -----
        This check is necessary before calling ensemble-wise operations
        like :meth:`ensemble_average` or :meth:`ensemble_std`, which
        require uniform time sampling across all series.
        
        Examples
        --------
        >>> multi = MultiTimeSeries(series_1khz, series_500hz)
        >>> multi.is_homogeneous()  # False - different sampling rates
        False
        >>> 
        >>> multi2 = MultiTimeSeries(series1_1khz, series2_1khz)
        >>> multi2.is_homogeneous()  # True - same sampling
        True
        '''
        dt = [x.value for x in self.delta_times(*args, **kwargs)]
        return len(set(dt)) == 1

    def _impersonateDeltaTime(self, *args, **kwargs):
        '''Assume the first delta time of the ones selected by the args'''
        dt = self.delta_times(*args, **kwargs)
        self.delta_time = dt[0]

    def ensemble_size(self):
        '''Returns the total ensemble size'''
        return sum([x.ensemble_size() for x in self._series])

    def delta_times(self, *args, **kwargs):
        '''Returns a vector of delta times'''

        # Known astropy bug (numpy < 1.17): units are lost when using hstack 
        # We remove them before stacking, and add them later
        dt = np.hstack( \
              [np.repeat(x.delta_time.to('s').value, x.ensemble_size())
               for x in self._series])

        dt = dt * u.s

        index = self.get_index_of(*args, **kwargs)
        if index is not None and len(index)>0:
            dt = dt[index]

        return dt

    @modify_help(arg_str='[time_idx]')
    def ensemble_average(self, *args, times=None, **kwargs):
        ''' Average across series at each sampling time '''
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().ensemble_average(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

    @modify_help(arg_str='[time_idx]')
    def ensemble_std(self, *args, times=None, **kwargs):
        ''' Standard deviation across series at each sampling time '''
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().ensemble_std(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

    @modify_help(arg_str='[time_idx]')
    def ensemble_median(self, *args, times=None, **kwargs):
        ''' Standard deviation across series at each sampling time '''
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().ensemble_median(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

# ___oOo___