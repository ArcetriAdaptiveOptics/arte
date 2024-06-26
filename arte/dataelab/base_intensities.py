import numpy as np
import astropy.units as u
from arte.dataelab.base_timeseries import BaseTimeSeries

class BaseIntensities(BaseTimeSeries):
    '''
    Time series for subaperture intensities
    '''
    def __init__(self, data, time_vector=None, astropy_unit=u.adu, data_label='Subaperture intensities', axes=None):
        super().__init__(data=data,
                         time_vector=time_vector,
                         astropy_unit=astropy_unit,
                         data_label=data_label,
                         axes=axes)

    def total_adu(self, threshold=0.1):
        '''
        Total number of ADU/frame. Optional threshold is relative to max
        '''
        data = self.time_average()
        data[np.where(data < data.max()*threshold)] = 0
        return data.sum()

