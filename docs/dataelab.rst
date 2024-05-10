Data elab
=========

For each of your data series, define a class derived from `BaseTimeseries`::

    from arte.dataelab.base_timeseries import BaseTimeseries

    class LBTSlopes(BaseTimeseries):
        def __init__(self, loader_or_data, time_vector=None):
            super().__init__(loader_or_data, time_vector)

Define a class derived from `AbstractFileNameWalker`,
defining the snapshot_dir() method and an additional method
for each file that you want to load::

    import os
    from pathlib import Path
    from arte.dataelab.base_file_walker import AbstractFileNameWalker

    class LBTFileNameWalker(AbstractFileNameWalker):
        def __init__(self):
            super().__init__()

        def snapshot_dir(self, tag):
            return Path('/data') / tag

        def slopes(self, tag):
            return self.snapshot_dir(tag) / 'slopes.fits'


Now define a class for the main analyzer, adding some access methods::

    from arte.dataelab.base_analyzer import BaseAnalyzer
    from arte.dataelab.data_loader import FitsDataLoader


    class LBTAnalyzer(BaseAnalyzer):
        def __init__(self, tag, recalc=False):
            super().__init__(tag, recalc=recalc)

            self._file_walker = LBTFileNameWalker()
            self._slopes = LBTSlopes(FitsDataLoader(self._file_walker.slopes()))

        def slopes(self):
            return self._slopes


In order to use the analyzer, rather than instantiate it directly, it is better
to use its :py:meth:`~arte.dataelab.base_analyzer.BaseAnalyzer.get` method. This method
has an internal cache and will reuse an Analyzer instance if the same tag
is requested multiple times::

    ee = LBTAnalyzer.get('20230303_112233')
    ee.slopes.get_data(times=[1*u.s, 2*u.s]) 



Submodules
----------

arte.dataelab.base_analyzer module
-----------------------------------

.. automodule:: arte.dataelab.base_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

arte.dataelab.base_timeseries module
-----------------------------------------

.. automodule:: arte.dataelab.base_timeseries
   :members:
   :undoc-members:
   :show-inheritance:

arte.dataelab.base_file_walker module
-------------------------------

.. automodule:: arte.dataelab.base_file_walker
   :members:
   :undoc-members:
   :show-inheritance:

