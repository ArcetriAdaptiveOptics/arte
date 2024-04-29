import numpy as np
import astropy.units as u

from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.time_series import Indexer
from arte.utils.show_array import show_array


signal_unit = u.def_unit('signal')


class BaseSlopes(BaseTimeSeries):
    '''Slopes recorded from a generic WFS'''

    def __init__(self, delta_time, loader_or_data, mapper2d=None, astropy_unit=signal_unit,
                 interleaved=True):
        super().__init__(delta_time,
                         loader_or_data=loader_or_data,
                         mapper2d=mapper2d,
                         astropy_unit=astropy_unit)
        self._interleaved = interleaved

    def get_index_of(self, *args, **kwargs):
        if self._interleaved:
            return Indexer().interleaved_xy(*args, **kwargs)
        else:
            return Indexer().sequential_xy(maxindex=self.ensemble_size(), *args, **kwargs)

    def get_display_sx(self):
        '''Raw slope-x data as a cube of 2d display'''
        sx = self.get_data('x')
        return self._mapper2d(sx)

    def get_display_sy(self):
        '''Raw slope-y data as a cube of 2d display'''
        sy = self.get_data('y')
        return self._mapper2d(sy)

    def get_display(self):
        '''Raw data as a cube of 2d display arrays'''
        sx2d = self.get_display_sx()
        sy2d = self.get_display_sy()
        return np.concatenate((sx2d, sy2d), axis=2)
    
    def imshow(self, cut_wings=0):
        '''
        Display X and Y slope 2d images
        cut_wings=x means that colorbar is saturated for array values below x percentile
        and above 100-x percentile. Default is 0, i.e. all data are displayed; values below
        0 are forced to 0, values above 50 are set to 50.
        '''
        title = "left:" + self._data_label + "-X, right:" + self._data_label + "-Y"
        array2show = self.get_display().mean(axis=0)
        return show_array(array2show, cut_wings, title, 'Subap', 'Subap', self._data_units)

    def vecshow(self):
        '''Display slopes as vector field'''
        sx2d = self.get_display_sx().mean(axis=0)
        sy2d = self.get_display_sy().mean(axis=0)

        import matplotlib.pyplot as plt
        plt.quiver(sx2d, sy2d)
        return plt
