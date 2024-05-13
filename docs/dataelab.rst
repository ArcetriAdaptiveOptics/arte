Data elab
=========


The elaboration library is based on the concept of *tag*: 
a name for a file collection corresponding to a snapshot
of the system state, contaning both istantaneus sampling of slow-varying data
(e.g. seeing measurements) and time series of fast varying data (pixel frames,
DM commands, etc), usually a few hundreds or thousands of samples each.
A tag is typically a timestamped directory containing
a list of FITS or numpy files for time series, and other text or binary files
for the rest of the data. The data inside a tag is typically saved on-demand
when an user requests it, or can be triggered automatically based on some event,
or even extracted from a more general telemetry store.

This library defines three main base classes acting in concert, plus
some smaller helper classes. You have to derive from each of the three
main base classes, adding the functionality needed for your custom data.

File walker
-----------

The file walker knows your directory structure and all your file names.
In general, it should know how to:
   * find the root directory of your data storage, based on your preferred
     mechanism (environment variables, configuration files, hard-coded paths, etc)
   * return the directory holding the data for a single tag, inside the root directory,
     including any internal structure if any (e.g. if the tags are divided by day)
   * return the full path for each single data type stored inside a tag

Define a class derived from :py:class:`~arte.dataelab.base_file_walker.AbstractFileNameWalker`,
defining the :py:meth:`~arte.dataelab.base_file_walker.AbstractFileNameWalker.snapshot_dir()` 
method and an additional method for each file that you want to load::

    import os
    from pathlib import Path
    from arte.dataelab.base_file_walker import AbstractFileNameWalker

    class LBTFileNameWalker(AbstractFileNameWalker):
        def __init__(self):
            super().__init__()

        def snapshot_dir(self, tag):
            # For this example we use the hard-coded "/data" path
            # You should use something more flexible like a configuration file
            # or an environment variable.
            return Path('/data') / tag

        def slopes(self, tag):
            return self.snapshot_dir(tag) / 'slopes.fits'


Time Series
-----------

For each of your data series (for example: pixel frames, slopes,
DM commands, etc), define a class derived from
:py:class:`~arte.dataelab.base_timeseries.BaseTimeSeries`::

    from arte.dataelab.base_timeseries import BaseTimeseries

    class LBTSlopes(BaseTimeseries):
        def __init__(self, loader_or_data, time_vector=None):
            super().__init__(loader_or_data, time_vector)

By default, the :py:meth:`~arte.time_series.time_series.TimeSeries.get_data`
method of this class will retrieve the entire data array. In order
to select a data subset, you should override the
:py:meth:`~arte.dataelab.base_timeseries.BaseTimeSeries.get_index_of`,
returning the index to be used to select a data subset (see the method
documentation for details).
What kind of arguments or keywords to accept is entirely up to you.
In this example we accept two keywords, 'x' and 'y', to return
the even and odd data members::

    from arte.dataelab.base_timeseries import BaseTimeseries

    class LBTSlopes(BaseTimeseries):
        def __init__(self, loader_or_data, time_vector=None):
            super().__init__(loader_or_data, time_vector)

        def get_index_of(self, *args, **kwargs):
            '''Return the x or y slopes'''
            if 'x' in kwargs:
                return slice(0, None)
            elif 'y' in kwargs:
                return slice(1, None)
            else:
                return None   # Get all data

Thanks to the base time series class, your one will inherit a huge amount
of useful methods. Please have a look at the :py:class:`~arte.dataelab.base_timeseries.BaseTimeSeries`
class documentation for a list.

Main analyzer
-------------

An Analyzer is a class that knows how to analyze all data inside a tag.
It will use a file walker instance in order get the file names, and
pass those to your custom data series classes to build the analyzer attributes.

Define your analyzer class deriving from the base analyzer, adding some access methods::

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

    # Get all x slopes between 1 and 2 seconds in this tag
    ee.slopes.get_data('x', times=[1*u.s, 2*u.s])


Interactive help
----------------

If you are unsure of how to navigate your data or which methods are available,
all Analyzer instances have an interactive help with sub-string search::

   >>> ee.help('std')
   
   ee.slopes.ensemble_std(*args[, times])  Standard deviation across series at each sampling time
   ee.slopes.time_std(*args[, times])      Standard deviation over time for each series


Help strings are automatically generated scanning the Analyzer member hiearchy, and
uses the first docstring line for each public method, so you can add your own
custom help as well.


Analyzer sets
-------------

*sets* are a list of Analyzer objects that can be addressed in a single call::

    from arte.dataelab.base_analyzer_set import AnalyzerSet

    file_walker = LBTFileNameWalker()
    LBTSetAnalyzer = AnalyzerSet(file_walker, class_type=LBTAnalyzer)

    # Get an entire day's worth of data
    myset = LBTSetAnalyzer('20230303_000000', '20230303_235959')

    # Get a list of slope rms, one value for each tag in the set.
    # Discard the first 0.1 seconds in each tag.
    slopes = myset.slopes.time_std(times=[0.1, None])


Disk cache
----------



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

