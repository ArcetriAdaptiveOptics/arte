import astropy.units as u

from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.time_series.indexer import Indexer
from arte.utils.unit_checker import separate_value_and_unit

signal_unit = u.def_unit('signal')



class BaseSlopes(BaseTimeSeries):
    """Wavefront sensor slopes time series.
    
    This class handles slopes (wavefront gradients) recorded from wavefront
    sensors. Slopes are stored as vectors where typically half the elements
    are x-slopes and half are y-slopes.
    
    By default, slopes are in xxxxyyyy order (all X slopes followed by all Y
    slopes). Individual slope components can be accessed using 'x' and 'y'
    selectors.
    
    Slopes can be remapped into 2D arrays representing the telescope pupil,
    where each subaperture position shows its measured slope value.
    
    Parameters
    ----------
    data : array_like or DataLoader
        Slope data with shape (nframes, nslopes) or (nslopes,)
    time_vector : array_like or DataLoader, optional
        Time vector for each frame
    astropy_unit : astropy.units.Unit, optional
        Physical unit for slope values (default: signal_unit)
    data_label : str, optional
        Label for plots (default: 'slopes')
    axes : sequence, optional
        Names for data axes
    
    Examples
    --------
    >>> slopes = BaseSlopes('slopes.fits')
    >>> sx = slopes.get_data('x')  # Get x-slopes only
    >>> sy = slopes.get_data('y')  # Get y-slopes only
    >>> slopes.imshow()  # Display as 2D pupil maps
    >>> slopes.vecshow()  # Display as vector field
    
    Notes
    -----
    Derived classes can customize the slope order by providing a different
    indexer implementation.
    """

    def __init__(self, data, time_vector=None, astropy_unit=signal_unit, data_label='slopes', axes=None):
        super().__init__(data=data,
                         time_vector=time_vector,
                         astropy_unit=astropy_unit,
                         data_label=data_label,
                         axes=axes)
        self._indexer = Indexer()

    def get_index_of(self, *args, **kwargs):
        return self._indexer.sequential_xy(self.ensemble_size(), *args, **kwargs)

    def imshow(self, cut_wings=0):
        '''
        Display X and Y slope 2d images
        cut_wings=x means that colorbar is saturated for array values below x percentile
        and above 100-x percentile. Default is 0, i.e. all data are displayed; values below
        0 are forced to 0, values above 50 are set to 50.
        '''
        title = "left:" + self._data_label + "-X, right:" + self._data_label + "-Y"
        super().imshow(cut_wings=cut_wings, title=title)

    def vecshow(self):
        '''Display slopes as vector field'''
        sx2d = self.get_display('x').mean(axis=0)
        sy2d = self.get_display('y').mean(axis=0)
        sx2d, _ = separate_value_and_unit(sx2d)
        sy2d, _ = separate_value_and_unit(sy2d)

        import matplotlib.pyplot as plt
        plt.quiver(sx2d, sy2d)
        return plt
