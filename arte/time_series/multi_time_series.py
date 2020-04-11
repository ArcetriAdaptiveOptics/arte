
import numpy as np
import astropy.units as u

from arte.utils.time_series import TimeSeries

class MultiTimeSeries(TimeSeries):
    '''
    This class is a container for multiple TimeSeries objects
    with potentially incompatible time sampling.
    '''

    def __init__(self, *args):

        super().__init__(None)

        self._series = []
        for v in args:
            self._series.append(v)

    def _get_not_indexed_data(self):
        return np.hstack([v._get_not_indexed_data() \
                          for v in self._series])
        
    def get_series(self, n):
        return self._series[n]

    def is_homogenous(self, *args, **kwargs):

        dt = [x.value for x in self.delta_times(args, **kwargs)]
        return len(set(dt)) == 1

    def __impersonateDeltaTime(self, *args, **kwargs):

        dt = self.delta_times(args, **kwargs)
        self.delta_time = dt[0]

    def ensemble_size(self):
        return sum([x.ensemble_size() for x in self._series])

    def delta_times(self, *args, **kwargs):

        # Known astropy bug: units are lost when using hstack 
        # We remove them before stacking, and add them later
        dt = np.hstack( \
              [np.repeat(x.delta_time.to('s').value, x.ensemble_size()) \
              for x in self._series])

        dt = dt * u.s

        index = self.get_index_of(*args, **kwargs)
        if index is not None and len(index)>0:
            dt = dt[index]

        return dt
             
    def time_average(self, times=None, *args, **kwargs):
        if times is None or self.is_homogenous(*args, **kwargs):
            self.__impersonateDeltaTime(*args, **kwargs) 
            return super().time_average(*args, **kwargs)
        else:
            raise Exception('Data series cannot be combined')
        
    def time_median(self, times=None, *args, **kwargs):
        if times is None or self.is_homogenous(*args, **kwargs):
            self.__impersonateDeltaTime(*args, **kwargs) 
            return super().time_median(*args, **kwargs)
        else:
            raise Exception('Data series cannot be combined')
        
    def time_std(self, times=None, *args, **kwargs):
        if times is None or self.is_homogenous(*args, **kwargs):
            self.__impersonateDeltaTime(*args, **kwargs) 
            return super().time_std(*args, **kwargs)
        else:
            raise Exception('Data series cannot be combined')
        
    def power(self, from_freq=None, to_freq=None,
              segment_factor=None, window='boxcar', *args, **kwargs):      

        if self.is_homogenous(*args, **kwargs):
            self.__impersonateDeltaTime(*args, **kwargs) 
            return super().power(from_freq, to_freq, segment_factor, \
                                 window, *args, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

