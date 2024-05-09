
import astropy.units as u
from arte.time_series.indexer import ModeIndexer
from arte.dataelab.base_projection import BaseProjection


class BaseResidualModes(BaseProjection):
    '''
    Base class for residual modes

    slopes_timeseries: time series object with slopes data
    modalrec: data object with modal reconstructor
    '''

    def __init__(self, slopes_timeseries, modalrec, astropy_unit=u.m, data_label='Modal coefficients'):
        super().__init__(slopes_timeseries,
                         modalrec,
                         astropy_unit=astropy_unit,
                         data_label=data_label)
        self._nmodes = None   # Lazy initialization
        self._indexer = None

    def nmodes(self):
        '''Number of modes'''
        if self._nmodes is None:
            self._nmodes = self._projection_matrix.get_data().shape[0]
        return self._nmodes

    def get_index_of(self, *args, **kwargs):
        if self._indexer is None:
            self._indexer = ModeIndexer(max_mode=self.nmodes())
        return self._indexer.modes(*args, **kwargs)

    def slopes(self):
        '''Slopes timeseries'''
        return self._source_timeseries

    def modalrec(self):
        '''Modal reconstructor'''
        return self._projection_matrix

# __oOo__