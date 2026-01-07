import numpy as np
import astropy.units as u

from arte.utils.help import add_help
from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import data_loader_factory
from arte.dataelab.unit_handler import UnitHandler
from arte.utils.show_array import show_array
from arte.time_series.axis_handler import AxisHandler


@add_help
class BaseData():
    """Generic static data container with lazy loading.
    
    BaseData represents static (non-time-series) data such as calibration
    matrices, reference frames, or configuration parameters. It supports:
    
    - Lazy loading from files or direct numpy arrays
    - Optional physical units via astropy.units
    - Automatic unit conversion
    - Integration with NotAvailable pattern for missing data
    
    Parameters
    ----------
    data : DataLoader, str, Path, or ndarray
        Data source. Can be:
        - DataLoader instance for lazy loading
        - Filename (string or Path) to load from
        - Numpy array for direct initialization
    astropy_unit : astropy.units.Unit, optional
        Physical unit for the data
    data_label : str, optional
        Human-readable label for plots (e.g., 'Interaction Matrix')
    axes : sequence, optional
        Names for data axes
    
    Attributes
    ----------
    _data_loader : DataLoader
        Internal data loader instance
    _astropy_unit : Unit
        Unit for the data
    _data_label : str
        Label for the data
    
    Examples
    --------
    >>> # Load from file
    >>> interaction_matrix = BaseData('imat.npy',
    ...     data_label='Interaction Matrix')
    >>> data = interaction_matrix.get_data()
    
    >>> # Direct initialization
    >>> reference = BaseData(np.zeros((512, 512)),
    ...     astropy_unit=u.nm,
    ...     data_label='Reference Frame')
    
    >>> # With lazy loading
    >>> from arte.dataelab.data_loader import NumpyDataLoader
    >>> loader = NumpyDataLoader('data.npy')
    >>> data = BaseData(loader)
    
    See Also
    --------
    BaseTimeSeries : For time-varying data
    DataLoader : Base class for lazy data loading
    """
    def __init__(self, data, astropy_unit=None, data_label=None, axes=None):
        
        data_loader = data_loader_factory(data, allow_none=False, name='data')

        try:
            # Test that the data file is there, when possible
            _ = data_loader.assert_exists()

        except AssertionError as e:
            print(e)
            NotAvailable.transformInNotAvailable(self)
            return
        
        if data_label is None:
            data_label = self.__class__.__name__

        self._data_loader = data_loader
        self._astropy_unit = astropy_unit
        self._data_label = data_label
        self._unit_handler = UnitHandler(wanted_unit = astropy_unit)
        self._axis_handler = AxisHandler(axes)

    def filename(self):
        '''Data filename (full path)'''
        return self._data_loader.filename()

    def get_data(self, *args, axes=None, **kwargs):
        '''Raw data, selecting elements based on the indexer'''

        raw_data = self._get_not_indexed_data()
        index = self.get_index_of(*args, **kwargs)
        if index is None:
            data = raw_data
        else:
            data = raw_data[index]
        return self._axis_handler.transpose(data, axes)

    def _get_not_indexed_data(self):
        return self._unit_handler.apply_unit(self._data_loader.load())

    def get_index_of(self, *args, **kwargs):
        '''Return data slice. Override in derived classes if needed'''
        return None

    @property
    def shape(self):
        return self.get_data().shape

    def data_label(self):
        '''Long-form data label (for plots)'''
        return self._data_label or ''

    def data_unit(self):
        '''Data unit string (for plots)'''
        return self._unit_handler.actual_unit_name() or ''

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._unit_handler.actual_unit()

    # Override to provide custom displays
    def _get_display_frame(self, data_to_display):
        return np.atleast_2d(data_to_display)

    def get_display(self, *args, **kwargs):
        '''Data mapped in 2d'''
        data_to_display = self.get_data(*args, **kwargs)
        display_data = self._get_display_frame(data_to_display)
        if isinstance(display_data, u.Quantity):
            return display_data.value
        else:
            return display_data

    def imshow(self, cut_wings=0, title='', xlabel='', ylabel='', **kwargs):
        '''
        Display a 2d image.

        cut_wings=x means that colorbar is saturated for array values below x percentile
        and above 100-x percentile. Default is 0, i.e. all data are displayed; values below
        0 are forced to 0, values above 50 are set to 50.
        '''
        if not title:
            title = self.data_label()
        axes = self._axis_handler.axes()
        if not ylabel and axes and len(axes) > 0:
            ylabel = axes[0]
        if not xlabel and axes and len(axes) > 1:
            xlabel = axes[1]
        array2show = self.get_display()
        return show_array(array2show, cut_wings, title, xlabel, ylabel, self.data_unit(), **kwargs)



# __oOo__