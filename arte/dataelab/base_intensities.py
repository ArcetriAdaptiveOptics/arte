import numpy as np
import astropy.units as u
from arte.dataelab.base_timeseries import BaseTimeSeries

class BaseIntensities(BaseTimeSeries):
    '''
    Time series for subaperture intensities
    '''
    def __init__(self, delta_time, loader_or_data, mapper2d=None, astropy_unit=u.adu):
        super().__init__(delta_time,
                         loader_or_data=loader_or_data,
                         mapper2d=mapper2d,
                         astropy_unit=astropy_unit)

    def total_adu(self, threshold=0.1):
        '''
        Total number of ADU/frame. Optional threshold is relative to max
        ''' 
        data = self.time_average()
        data[np.where(data < data.max()*threshold)] = 0    
        return data.sum() * self._astropy_unit

