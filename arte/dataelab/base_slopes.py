import astropy.units as u

from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.time_series.indexer import Indexer
from arte.utils.show_array import show_array
from arte.utils.unit_checker import separate_value_and_unit

signal_unit = u.def_unit('signal')



class BaseSlopes(BaseTimeSeries):
    '''Slopes recorded from a generic WFS'''

    def __init__(self, loader_or_data, time_vector=None, astropy_unit=signal_unit, data_label='slopes',
                 interleaved=True):
        super().__init__(loader_or_data=loader_or_data,
                         time_vector=time_vector,
                         astropy_unit=astropy_unit,
                         data_label=data_label)
        # Special display handling
        self._interleaved = interleaved
        self._indexer = Indexer()

    def get_index_of(self, *args, **kwargs):
        if self._interleaved:
            return self._indexer.interleaved_xy(*args, **kwargs)
        else:
            return self._indexer.sequential_xy(maxindex=self.ensemble_size(), *args, **kwargs)

    def imshow(self, cut_wings=0):
        '''
        Display X and Y slope 2d images
        cut_wings=x means that colorbar is saturated for array values below x percentile
        and above 100-x percentile. Default is 0, i.e. all data are displayed; values below
        0 are forced to 0, values above 50 are set to 50.
        '''
        title = "left:" + self._data_label + "-X, right:" + self._data_label + "-Y"
        array2show = self.get_display().mean(axis=0)
        return show_array(array2show, cut_wings, title, 'Subap', 'Subap', self.data_unit())

    def vecshow(self):
        '''Display slopes as vector field'''
        sx2d = self.get_display('x').mean(axis=0)
        sy2d = self.get_display('y').mean(axis=0)
        sx2d, _ = separate_value_and_unit(sx2d)
        sy2d, _ = separate_value_and_unit(sy2d)

        import matplotlib.pyplot as plt
        plt.quiver(sx2d, sy2d)
        return plt
