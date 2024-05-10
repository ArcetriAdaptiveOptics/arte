from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.utils.not_available import NotAvailable
from arte.dataelab.cache_on_disk import cache_on_disk
from arte.dataelab.data_loader import OnTheFlyLoader

class BaseProjection(BaseTimeSeries):

    def __init__(self, source_timeseries, projection_matrix, astropy_unit=None, data_label=None):
        try:
            assert not isinstance(source_timeseries, NotAvailable)
            assert not isinstance(projection_matrix, NotAvailable)
            super().__init__(loader_or_data=OnTheFlyLoader(self.project),
                             time_vector=OnTheFlyLoader(source_timeseries.get_time_vector),
                             astropy_unit=astropy_unit,
                             data_label=data_label)
            self._unit_handler.set_force(True)

        except AssertionError as e:
            NotAvailable.transformInNotAvailable(self)
            return
        self._source_timeseries = source_timeseries
        self._projection_matrix = projection_matrix

    @cache_on_disk
    def project(self):
        '''Perform projection'''
        data = self._source_timeseries.get_data()
        proj = self._projection_matrix.get_data()
        print(data, proj)
        if isinstance(data, NotAvailable) or isinstance(proj, NotAvailable):
            data = NotAvailable()
        else:
            data = data @ proj
        return data

# __oOo__
