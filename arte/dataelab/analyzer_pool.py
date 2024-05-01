
class AnalyzerPool(object):
    '''
    An AnalyzerPool mantains a memory cache of Analyzer objects.
    The get() method returns a a cached version if available

    *type_* is the class type for the Analyzer instance
    (e.g. BaseAnalyzer, or a derived instance)
    '''
    def __init__(self, logger, configuration, type_):
        self._logger = logger
        self._type = type_
        self._configuration = configuration
        self._cache = dict()

    def get(self, snapshot_tag, recalc=False):
        '''
        Return an Analyzer object corresponding to *snapshot_tag*,
        possibly cached from this memory pool.
        '''
        if snapshot_tag not in self._cache:
            result = self._type(
                snapshot_tag, self._configuration, self._logger, recalc=recalc)
            self._cache[snapshot_tag] = result
        elif recalc:
            self._cache[snapshot_tag].recalc() 
        return self._cache[snapshot_tag]

    def delete(self, snapshot_tag):
        '''Delete the Analyzer object corresponding to *snapshot_tag* from the memory cahce'''
        if snapshot_tag in self._cache:
            del self._cache[snapshot_tag]

