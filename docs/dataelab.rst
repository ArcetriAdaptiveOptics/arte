Data elab
=========

This library is intended for analysis of laboratory-generated datasets
with the following charateristics:
* grouped into discrete sets
* each set contains a snapshot of the system state, plus one or more time series 
of fast-varying data (pixel frames, DM commands, etc),
usually a few hundreds or thousands of samples each, stored in numpy, FITS
or other numeric formats.

Tags
----

The elaboration library is based on the concept of *tag*: 
a name for a file collection corresponding to a snapshot
of the system state, contaning both istantaneus sampling of slow-varying data
(e.g. seeing measurements) and time series of fast-varying data (pixel frames,
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

Derive your FileWalker class from :py:class:`~arte.dataelab.base_file_walker.AbstractFileNameWalker`,
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
        def __init__(self, data, time_vector=None, astropy_unit=None):
            super().__init__(data, time_vector, astropy_unit)

By default, the :py:meth:`~arte.time_series.time_series.TimeSeries.get_data`
method of this class will retrieve the entire data array. In order
to select a data subset, you should override the
:py:meth:`~arte.dataelab.base_timeseries.BaseTimeSeries.get_index_of`,
returning the index to be used to select a data subset (see the method
documentation for details).
What kind of arguments or keywords to accept is entirely up to you.
In this example we accept two arguments, 'x' and 'y', to select
the even and odd data members::

    from arte.dataelab.base_timeseries import BaseTimeseries

    class LBTSlopes(BaseTimeseries):
        def __init__(self, data, time_vector=None):
            super().__init__(data, time_vector)

        def get_index_of(self, *args, **kwargs):
            '''Select the x or y slopes'''
            if 'x' in args:
                return slice(0, None)
            elif 'y' in args:
                return slice(1, None)
            else:
                return None   # Get all data

Thanks to the base time series class, your one will inherit a number
of useful methods. Please have a look at the :py:class:`~arte.dataelab.base_timeseries.BaseTimeSeries`
class documentation for a list.

Data Loaders
------------

Data loaders enable the elaboration library to use *lazy loading*, where the
data is only loaded when needed and not before. Thus, instead of initializing
time series with a raw numpy array, you should instead pass them data loaders.

The simplest data loader is just a standard Python function that will return
data. The `~arte.dataelab.base_timeseries.BaseTimeSeries`
will call this function only when data is accessed and not before.
Any Python callable will work this way.

A data loader instance derived from `~arte.dataelab.data_loader.DataLoader`
allows one more lazy check: if the data loaded is from a file, the loader
*exists()* method allows the time series to check whether the file exist,
and thus stop early in case of missing files.
 
For numpy (\*.npy and \*.npz) and FITS files you can use one of the built-ins
`~arte.dataelab.data_loader.NumpyDataLoader` and
`~arte.dataelab.data_loader.FitsDataLoader`. Otherwise, you can
implement your own deriving from the abstract `~arte.dataelab.data_loader.DataLoader` class.

The numpy data loader can automatically handle compressed numpy files (\*.npz).
Since \*.npz files work like dictionaries, the optional *key* argument is needed
to specify which array from the archive must be loaded. For example::

    from arte.dataelab.base_timeseries import BaseTimeseries
    from arte.dataelab.data_loader import NumpyDataLoader

    class LBTSlopes(BaseTimeseries):
        def __init__(self, filename, time_vector=None, astropy_unit=None):
            loader = NumpyDataLoader(filename, key='slopes')
            super().__init__(loader, time_vector, astropy_unit)

As seen in this example, the loader instance is passed in place of the data.

Main analyzer
-------------

An Analyzer is a class that knows how to analyze all data inside a tag.
It will use a file walker instance in order get the file names, and
pass those to your custom data series classes to build the analyzer attributes.

Derive your analyzer class from the base analyzer, adding some access methods::

    from arte.dataelab.base_analyzer import BaseAnalyzer
    from arte.dataelab.data_loader import FitsDataLoader


    class LBTAnalyzer(BaseAnalyzer):
        def __init__(self, tag, recalc=False):
            super().__init__(tag, recalc=recalc)

            self._file_walker = LBTFileNameWalker()
            self._slopes = LBTSlopes(FitsDataLoader(self._file_walker.slopes()))

        def slopes(self):
            return self._slopes


Note how we use a :py:class:`~arte.dataelab.data_loader.FitsDataLoader` instance
instead of passing the filename directly. The filename would work too, but a data loader
allows the BaseTimeSeries class to employ lazy loading while still doing things
at construction time like checking for the file existance, etc. There is a
:py:class:`~arte.dataelab.data_loader.NumpyDataLoader` for numpy files as well.

Usually the time series sclass will correspond to a single FITS or numpy file,
or to a subsection (like a single FITS extension). Time (or just the sample index)
is supposed to be the first axis. If your data layout is different, check if the
*transpose_axes* option of :py:class:`~arte.dataelab.data_loader.FitsDataLoader`
and :py:class:`~arte.dataelab.data_loader.NumpyDataLoader` is enough.
If you have a more exotic data layout
you can either do the work manually, building a numpy array
and passing it to the constructor, or derive your own class from
:py:class:`~arte.dataelab.data_loader.DataLoader` for a more structured approach.

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
use the first docstring line for each public method, so you can add your own
custom help as well.


Analyzer sets
-------------

*sets* are a list of Analyzer objects that can be addressed in a single call,
based on a list of tags or just the first and last one. The base class
needs to be initialized with your file walker and the analyzer class type
(note, just the type, not a instance)::

    from arte.dataelab.base_analyzer_set import BaseAnalyzerSet

    class LBTAnalyzerSet(BaseAnalyzerSet):
        def __init__(self, first, last, recalc=False):
            super().__init___(first, last, recalc=recalc,
                              file_walker=LBTFileNameWalker(),
                              analyzer_type=LBTAnalyzer)

    # Get an entire day's worth of data
    myset = LBTSetAnalyzer('20230303_000000', '20230303_235959')

    # Get a list of slope rms, one value for each tag in the set.
    # Discard the first 0.1 seconds in each tag.
    slopes = myset.slopes.time_std(times=[0.1, None])

In order for the set to work, you have to add another method to your custom FileNameWalker
(of course replace the example algorithm here with the correct one for you)::

        class LBTFileNameWalker(AbstractFileNameWalker):
        
            ...

            def find_tag_between_dates(self, tag_start, tag_stop):
                '''Return all tags between *tag_start* and *tag_stop* '''
                tags = os.listdir(mydir)
                return list(filter(lambda x: tag_start <= x <= tag_stop, tags))

the :py:meth:`~arte.dataelab.base_file_walker.AbstractFileNameWalker.find_tag_between_dates`
method must return a list of tags between the two extremes

Disk cache
----------

TODO

Not available
-------------

TODO

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

arte.dataelab.data_loader module
-------------------------------

.. automodule:: arte.dataelab.data_loader
   :members:
   :undoc-members:
   :show-inheritance:

