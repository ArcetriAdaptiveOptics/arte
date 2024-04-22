import numpy as np
from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.utils.not_available import NotAvailable
from arte.utils.cache_on_disk import cache_on_disk
from arte.dataelab.data_loader import OnTheFlyLoader


class BaseProjection(BaseTimeSeries):

    def __init__(self, source_timeseries, projection_matrix, mapper2d=None, astropy_unit=None):
        try:
            assert not isinstance(source_timeseries, NotAvailable)
            assert not isinstance(projection_matrix, NotAvailable)
            super().__init__(source_timeseries.delta_time,
                             loader=OnTheFlyLoader(self.project),
                             mapper2d=mapper2d,
                             astropy_unit=astropy_unit)
        except Exception as e:
            NotAvailable.transformInNotAvailable(self)
            return
        self._source_timeseries = source_timeseries
        self._projection_matrix = projection_matrix

    @cache_on_disk
    def project(self):
        data = self._source_timeseries.get_data()
        proj = self._projection_matrix.get_data()
        print(data.shape, proj.shape)
        print(data, proj)
        if isinstance(data, NotAvailable) or isinstance(proj, NotAvailable):
            data = NotAvailable()
        else:
            data = data @ proj
        return data


