
import numpy as np
import astropy.units as u
from arte.dataelab.base_timeseries import BaseTimeSeries

class BasePixels(BaseTimeSeries):
    '''
    Time series for pixel data
    '''
    def __init__(self, loader_or_data, time_vector=None, mapper2d=None, astropy_unit=u.adu, data_label='Pixel values'):
        super().__init__(loader_or_data=loader_or_data,
                         time_vector=time_vector,
                         mapper2d=mapper2d,
                         astropy_unit=astropy_unit,
                         data_label=data_label)

    def _get_display_cube(self, data):
        '''3d numpy array with pixel image over time'''
        npix = np.sqrt(data.shape[1])
        if npix != int(npix):
            raise Exception('Cannot remap pixels into 2d, frame is not square')
        npix = int(npix)
        return data.reshape((data.shape[0], npix, npix))

    def total_adu(self, threshold=0.1):
        '''Total number of ADU/frame. Optional threshold is relative to max.'''
        data = self.time_average()
        data[np.where(data < data.max()*threshold)] = 0
        return data.sum()

# __oOo__
