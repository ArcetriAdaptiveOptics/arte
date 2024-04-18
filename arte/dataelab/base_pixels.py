
import numpy as np
import astropy.units as u
from arte.dataelab.base_timeseries import BaseTimeSeries

class BasePixels(BaseTimeSeries):
    '''
    Time series for pixel data
    '''
    def __init__(self, delta_time, loader, mapper2d=None, astropy_unit=u.adu):
        BaseTimeSeries.__init__(delta_time,
                                loader=loader,
                                mapper2d=mapper2d,
                                astropy_unit=astropy_unit)

    def get_display(self):
        '''3d numpy array with pixel image over time'''
        p = self.get_data()
        npix = np.sqrt(p.shape[1])
        if npix != int(npix):
            raise Exception('Cannot remap pixels into 2d, frame is not square')
        npix = int(npix)
        return p.reshape((p.shape[0], npix, npix))

    def total_adu(self, threshold=0.1):
        '''Total number of ADU/frame. Optional threshold is relative to max.'''
        data = self.time_average()
        data[np.where(data < data.max()*threshold)] = 0    
        return data.sum() * self._astropy_unit

# __oOo__