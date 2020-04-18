
import numpy as np
import astropy.units as u

from arte.time_series import TimeSeries

class MultiTimeSeries(TimeSeries):
    '''
    Join multiple TimeSeries objects with incompatible time sampling.
    
    After being initialized with a list of TimeSeries or objects, which can
    have different ensemble sizes and length, it behaves as if all the data
    was part of a single ensemble. Operations that make calculations
    time-wise (like time_std()) work in any case. The ensemble-wise ones
    (like ensemble_std()) only work if deltaTime is the same
    across all series, and raise an Exception otherwise.
    '''

    def __init__(self, *args):

        super().__init__(None)

        self._series = []
        for v in args:
            self.add_series(v)

    def add_series(self, series):
        self._series.append(series)

    def _get_not_indexed_data(self):
        return np.hstack([v._get_not_indexed_data()
                          for v in self._series])
        
    def get_series(self, n):
        return self._series[n]

    def is_homogeneous(self, *args, **kwargs):

        dt = [x.value for x in self.delta_times(*args, **kwargs)]
        return len(set(dt)) == 1

    def __impersonateDeltaTime(self, *args, **kwargs):

        dt = self.delta_times(*args, **kwargs)
        self.delta_time = dt[0]

    def ensemble_size(self):
        return sum([x.ensemble_size() for x in self._series])

    def delta_times(self, *args, **kwargs):

        # Known astropy bug: units are lost when using hstack 
        # We remove them before stacking, and add them later
        dt = np.hstack( \
              [np.repeat(x.delta_time.to('s').value, x.ensemble_size())
               for x in self._series])

        dt = dt * u.s

        index = self.get_index_of(*args, **kwargs)
        if index is not None and len(index)>0:
            dt = dt[index]

        return dt
             
    def ensemble_average(self, *args, **kwargs):
        if self.is_homogeneous(*args, **kwargs):
            self.__impersonateDeltaTime(*args, **kwargs) 
            return super().time_average(*args, **kwargs)
        else:
            raise Exception('Data series cannot be combined')
        
    def ensemble_std(self, *args, **kwargs):
        if self.is_homogeneous(*args, **kwargs):
            self.__impersonateDeltaTime(*args, **kwargs) 
            return super().time_std(*args, **kwargs)
        else:
            raise Exception('Data series cannot be combined')
        
# ___oOo___