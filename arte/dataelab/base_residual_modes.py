
import astropy.units as u
from arte.time_series.indexer import Indexer
from arte.dataelab.base_projection import BaseProjection


class BaseResidualModes(BaseProjection):
    '''
    Base class for residual modes

    slopes_timeseries: time series object with slopes data
    modalrec: data object with modal reconstructor
    '''

    def __init__(self, slopes_timeseries, modalrec, astropy_unit=u.m):
        BaseProjection.__init__(slopes_timeseries,
                                modalrec,
                                astropy_unit=astropy_unit)
        self._nmodes = None   # Lazy initialization

    def nmodes(self):
        '''Number of modes'''
        if self._nmodes is None:
            self._nmodes = self._modalrec().get_data().shape[0]
        return self._nmodes

    def get_index_of(self, *args, **kwargs):
        return Indexer.modes(*args, max_mode=self.nmodes(), **kwargs)

    def slopes(self):
        return self._source_timeseries

    def modalrec(self):
        return self._projection_matrix

# __oOo__