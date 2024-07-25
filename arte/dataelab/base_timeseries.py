import numbers
from functools import cached_property, cache
import numpy as np
from astropy import units as u

from arte.time_series.time_series import TimeSeries
from arte.utils.help import modify_help
from arte.utils.not_available import NotAvailable
from arte.dataelab.data_loader import data_loader_factory, data_axes
from arte.dataelab.unit_handler import UnitHandler
from arte.dataelab.dataelab_utils import is_dataelab
from arte.time_series.indexer import DefaultIndexer
from arte.utils.displays import movie, tile, savegif
from arte.utils.show_array import show_array

class BaseTimeSeries(TimeSeries):
    '''
    Generic time series.

    A time series holds:
        - N samples of numeric data of arbitrary type and dimensions, with an optional astropy unit.
        - sampling time for each sample
        - optionally, a data label for plots and displays

    Data is accessed with the get_data() method. Derived classes can define specialized
    arguments to return data subsets, e.g. get_data(quadrant=2)
    Basic arithmetic operations (sum, div, etc) are supported. Astropy units, if present,
    will be enforced.

    Parameters
    ----------
    data_loader: instance of DataLoader or derived class, or numpy array, or filename (string or pathlib instance)
         time series data [time, data_d0 [, data_d1...]]
    time_vector: instance of DataLoader or derived class, or numpy_array, or None
         time vector data. If None, a default counter from 0 to N-1 samples will be assigned
    astropy_unit: astropy unit or None
         if possible, astropy unit to use with the data.
    data_label: string or None
         human-readable label for plot (e.g.: "Surface modal coefficients" )
    axes: sequence or None
         sequence of axes names, optional
    '''
    def __init__(self, data, time_vector=None, astropy_unit=None, data_label=None, axes=None):

        data_loader = data_loader_factory(data, allow_none=False, name='data')
        time_vector_loader = data_loader_factory(time_vector, allow_none=True, name='time_vector')
        if axes is None:
            # Try to derive from input data
            axes = data_axes(data)

        try:
            super().__init__(axes=axes)

            # Also test that the data file is there, when possible
            _ = data_loader.assert_exists()
            if time_vector_loader is not None:
                _ = time_vector_loader.assert_exists()

        except AssertionError as e:
            print(e)
            NotAvailable.transformInNotAvailable(self)
            return

        self._data_loader = data_loader
        self._time_vector_loader = time_vector_loader
        self._astropy_unit = astropy_unit
        self._data_label = data_label
        self._unit_handler = UnitHandler(wanted_unit = astropy_unit)
        self._default_indexer = DefaultIndexer()

    def filename(self):
        '''Data filename (full path)'''
        return self._data_loader.filename()

    @cached_property
    def shape(self):
        '''Data series shape

        TODO depending on the data loader, it could be found
        without loading the entire data array
        '''
        return self.get_data().shape

    @cache
    def _get_not_indexed_data(self):
        '''Lazy loading of raw data and astropy unit application'''
        return self._unit_handler.apply_unit(self._data_loader.load())

    def _get_time_vector(self):
        '''Lazy loading of time vector'''
        if self._time_vector_loader is not None:
            return self._time_vector_loader.load()
        else:
            return super()._get_time_vector()

    def get_index_of(self, *args, **kwargs):
        '''Return a selection index

        Parameters
        ----------
        *args: tuple
           user-defined arguments for data selection
        **kwargs: dict, optional
           extra user-defined arguments for data selection

        Returns
        -------
        index: None, integer, list of integers, slice objects, or a tuple of the previous possibilities.
           indexes to select a data subset. If None, all data is selected.
           An integer, a list of integers or a slice will select those rows.
           If the data is multidimensional, a tuple can be returned where each element
           will select data across a single dimension.
           For a detailed explanation, see the "Advanced indexing" topic at
           https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
        '''
        return self._default_indexer.elements(*args, **kwargs)

    def data_label(self):
        return self._data_label

    def data_unit(self):
        '''Data unit string (for plots)'''
        return self._unit_handler.actual_unit_name()

    def astropy_unit(self):
        '''Data unit as an astropy unit'''
        return self._unit_handler.actual_unit()

    # Override to provide custom displays
    def _get_display_cube(self, data_to_display):
        '''Generate a 3d cube for display'''
        return np.atleast_3d(data_to_display)

    @modify_help(arg_str='[series_idx], [times=[from, to]]')
    def get_display(self, *args, times=None, **kwargs):
        '''Display cube for the specified time interval'''
        data_to_display = self.get_data(*args, times=times, **kwargs)
        display_data = self._get_display_cube(data_to_display)
        if isinstance(display_data, u.Quantity):
            return display_data.value
        else:
            return display_data
    
    def get_display_axes(self):
        '''Display cube axes names as a list of 3 strings'''
        return ('time', '', '')

    @modify_help(arg_str='[series_idx], [times=[from, to]]')
    def movie(self, *args, interval=0.1, **kwargs):
        '''Display data as a movie'''
        frames = self.get_display(*args, **kwargs)
        movie(frames, interval=interval)

    @modify_help(arg_str='[series_idx], [times=[from, to]]')
    def tile(self, *args, rowlength=10, **kwargs):
        '''Display data as a tiled 2d frame'''
        frames = self.get_display(*args, **kwargs)
        return tile(frames, rowlength=rowlength)

    @modify_help(arg_str='filename, [series_idx], [times=[from, to]]')
    def savegif(self, filename, *args, interval=0.1, loop=0, **kwargs):
        '''Save data as an animated GIF'''
        frames = self.get_display(*args, **kwargs)
        savegif(frames, filename, interval=interval, loop=loop)

    def _check_broadcast(self, other):
        _ = np.broadcast_shapes(self.shape, other.shape)

    def _check_mult(self, other):
        if not self.shape[1] == other.shape[0]:
            raise ValueError(f'Data dimensions do not match: {self.shape} and {other.shape}')

    def _check_none(self, other):
        return None

    def _check_and_execute(self, check, other, func):
        check(other)
        return func(self.get_data(), other.get_data())

    def _operator(self, other, func, check):
        if isinstance(other, (numbers.Number, u.Quantity, np.ndarray)):
            new_data = func(self.get_data(), other)
        elif not is_dataelab(other):
            return NotImplemented
        else:
            new_data = lambda: self._check_and_execute(check, other, func)

        if self.__class__ == other.__class__:
            new_class = self.__class__
        else:
            new_class = BaseTimeSeries
        return new_class(new_data, time_vector=self._get_time_vector,
                         astropy_unit=None,
                         data_label=self._data_label)

    def __add__(self, other):
        return self._operator(other, lambda x, y: x + y, self._check_broadcast)

    def __sub__(self, other):
        return self._operator(other, lambda x, y: x - y, self._check_broadcast)

    def __mul__(self, other):
        return self._operator(other, lambda x, y: x * y, self._check_broadcast)

    def __truediv__(self, other):
        return self._operator(other, lambda x, y: x / y, self._check_broadcast)

    def __floordiv__(self, other):
        return self._operator(other, lambda x, y: x // y, self._check_broadcast)

    def __mod__(self, other):
        return self._operator(other, lambda x, y: x % y, self._check_broadcast)

    def __matmul__(self, other):
        return self._operator(other, lambda x, y: x @ y, self._check_mult)

    def __pow__(self, other):
        return self._operator(other, lambda x, y: x ** y, self._check_broadcast)

    def __neg__(self):
        return self._operator(self, lambda x, y: -x, self._check_none)

    def __abs__(self):
        return self._operator(self, lambda x, y: abs(x), self._check_none)

    def imshow(self, *args, cut_wings=0, title='', xlabel='', ylabel='', **kwargs):
        '''
        Display a 2d image.

        cut_wings=x means that colorbar is saturated for array values below x percentile
        and above 100-x percentile. Default is 0, i.e. all data are displayed; values below
        0 are forced to 0, values above 50 are set to 50.
        '''
        array2show = self.get_display(*args, **kwargs).mean(axis=0)
        _, default_xlabel, default_ylabel = self.get_display_axes()
        if not xlabel:
            xlabel = default_xlabel
        if not ylabel:
            ylabel = default_ylabel
        if not title:
            title = self.data_label()

        return show_array(array2show, cut_wings, title, xlabel, ylabel, self.data_unit())

    def plot(self, *args, cut_wings=0, title='', xlabel='', ylabel='', **kwargs):
        '''
        Plot timeseries
        '''
        import matplotlib.pyplot as plt

        data = self.get_data(*args, **kwargs)
        for index in range(data.shape[1]):
            plt.plot(data[:,index])
        return plt
            
    @modify_help(call='plot_hist([series_idx], from_freq=xx, to_freq=xx, )')
    def plot_hist(self, *args, from_t=None, to_t=None,
                  overplot=None, plot_to=None,
                  label=None,  **kwargs):
        '''Plot histogram. TODO: does not work, rewrite'''
        hist = self.get_data(*args, **kwargs)
        t = self._get_time_vector()
        if from_t is None: from_t=t.min()
        if to_t is None: to_t=t.max()
        ul = t <= to_t
        dl = t >= from_t
        lim = ul & dl
        t = t[lim]
        hist = hist[lim]

        from matplotlib.axes import Axes
        if plot_to is None:
            import matplotlib.pyplot as plt
        else:
            plt = plot_to
        if not overplot:
            plt.cla()
            plt.clf()

        lines = plt.plot(t, hist)
        if self.data_unit() is None:
            units = "(a.u.)"
        else:
            units = self.data_unit()

        title = 'Time history'
        xlabel = 'time [s]'
        ylabel = '['+units+']'
        if self.data_label() is not None:
            title = title + " of " + self.data_label()
            ylabel = self.data_label() + " " + ylabel

        if isinstance(plt, Axes):
            plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if label is not None:
                if isinstance(label, str):
                    plt.legend([label] * len(lines))
                else:
                    plt.legend(label)
        return plt

    @modify_help(call='plot_spectra([series_idx], from_freq=xx, to_freq=xx)')
    def plot_spectra(self, *args, from_freq=None, to_freq=None,
                     segment_factor=None,
                     overplot=False,
                     label=None, plot_to=None,
                     lineary=False, linearx=False,
                     **kwargs):
        '''Plot PSD'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cut_frequency()

        # linearx=True forces lineary to True
        if linearx: lineary=True
        if lineary and not linearx:
            p_shape = power.shape
            if len(p_shape)== 1:
                power *= freq
            else:
                power *= np.repeat(freq.reshape(p_shape[0],1), p_shape[1], axis=1)

        from matplotlib.axes import Axes

        if plot_to is None:
            import matplotlib.pyplot as plt
        else:
            plt = plot_to

        if not overplot:
            plt.cla()
            plt.clf()
        lines = plt.plot(freq[1:], power[1:])

        if lineary and not linearx:
            plt.semilogx()
        elif linearx and not lineary:
            plt.semilogy()
        elif not (linearx and lineary):
            plt.loglog()

        if self.data_unit() is None:
            units = "(a.u.)"
        else:
            units = self.data_unit()

        title = 'PSD'
        if self.data_label() is not None:
            title = title + " of " + self.data_label()
        xlabel = 'frequency [Hz]'
        if lineary and not linearx:
            ylabel = 'frequency*PSD ['+units+'^2]'
        else:
            ylabel = 'PSD ['+units+'^2]/Hz'
        if isinstance(plt, Axes):
            plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if label is not None:
                if isinstance(label, str):
                    plt.legend([label] * len(lines))
                else:
                    plt.legend(label)
        return plt

    @modify_help(call='plot_cumulative_spectra([series_idx], from_freq=xx, to_freq=xx)')
    def plot_cumulative_spectra(self, *args, from_freq=None, to_freq=None,
                                segment_factor=None,
                                label=None, plot_to=None,
                                overplot=False, plot_rms=False, lineary=False,
                                **kwargs):
        '''Plot cumulative PSD'''
        power = self.power(*args, from_freq=from_freq, to_freq=to_freq,
                           segment_factor=segment_factor,
                           **kwargs)
        freq = self.last_cut_frequency()
        freq_bin = freq[1]-freq[0] # frequency bin

        from matplotlib.axes import Axes

        if plot_to is None:
            import matplotlib.pyplot as plt
        else:
            plt = plot_to

        if not overplot:
            plt.cla()
            plt.clf()

        # cumulated PSD
        cumpsd = np.cumsum(power, 0) * freq_bin
        if plot_rms:
            # cumulated RMS
            cumpsd = np.sqrt(cumpsd)
            label_str = "RMS"
            label_pow = ""
        else:
            # cumulated PSD
            label_str = "Variance"
            label_pow = "^2"

        lines = plt.plot(freq[1:], cumpsd[1:])
        if lineary:
            plt.semilogx()
        else:
            plt.loglog()

        if self.data_unit() is None:
            units = "(a.u.)"
        else:
            units = self.data_unit()

        title = "Cumulated " + label_str
        if self.data_unit() is not None:
            title = title + " of " + self.data_label()
        xlabel = 'frequency [Hz]'
        ylabel = 'Cumulated '+label_str+' ['+units+label_pow+']'
        if isinstance(plt, Axes):
            plt.set(title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if label is not None:
                if isinstance(label, str):
                    plt.legend([label] * len(lines))
                else:
                    plt.legend(label)
        return plt



# __oOo__
