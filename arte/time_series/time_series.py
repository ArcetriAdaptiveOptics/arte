import abc
import numpy as np
from functools import cached_property
from scipy.signal import welch
from arte.utils.not_available import NotAvailable
from arte.utils.help import add_help, modify_help
from arte.utils.iterators import pairwise
from arte.utils.unit_checker import make_sure_its_a
from astropy import units as u


@add_help
class TimeSeries(metaclass=abc.ABCMeta):
    '''
    Base class for implementing operations on data representing time series.

    Derived classes must implement a `_get_not_indexed_data()` method
    that returns a numpy array of shape (n_time_elements, n_ensemble_elements).

    Derived classes must also implement a `get_index_of()` method to add
    ensemble indexing with arbitrary *args and **kwargs parameters
    (e.g. returning of partial subset based on indexes or names).

    Originally implemented as part of the ARGOS codebase.
    '''

    def __init__(self):
        self._frequency = None
        self._lastCuttedFrequency = None
        self._power = None
        self._segment_factor = None
        self._window = None

    @abc.abstractmethod
    def _get_not_indexed_data(self):
        pass

    def _get_time_vector(self):
        '''Override to provide a custom time vector'''
        return np.arange(len(self._get_not_indexed_data()))

    def get_data(self, *args, times=None, **kwargs):
        '''Raw data as a matrix [time, series]'''

        data = self._get_not_indexed_data()
        if times is not None:
            time_vector = self._get_time_vector()
            if isinstance(time_vector, u.Quantity):
                start = make_sure_its_a(time_vector.unit, times[0])
                stop = make_sure_its_a(time_vector.unit, times[1])
            else:
                start, stop = times
            idxs = np.logical_and(time_vector >= start, time_vector < stop)
            data = data[idxs]
        index = self.get_index_of(*args, **kwargs)
        if index is not None:
            data = data[:, index]
        return data

    @abc.abstractmethod
    def get_index_of(self, *args, **kwargs):
        pass

    def data_units(self):
        '''Override to return a string with a compact unit notation'''
        return None
    
    def data_label(self):
        '''Override to return a string with a readable unit name'''
        return None

    @cached_property
    def delta_time(self):
        '''Property with the interval between samples'''
        time_vector = self._get_time_vector()
        return time_vector[1] - time_vector[0]

    def frequency(self):
        return self._frequency

    def time_vector(self):
        return self._get_time_vector()

    def last_cutted_frequency(self):
        return self._lastCuttedFrequency

    def ensemble_size(self):
        '''Number of distinct series in this time ensemble'''
        not_indexed_data = self._get_not_indexed_data()
        return not_indexed_data.shape[1]

    def time_size(self):
        '''Number of time samples in this time ensemble'''
        not_indexed_data = self._get_not_indexed_data()
        return not_indexed_data.shape[0]

    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def power(self, from_freq=None, to_freq=None,
              segment_factor=None, window='boxcar', *args, **kwargs):
        '''Power Spectral Density across specified series'''

        index = self.get_index_of(*args, **kwargs)
        if segment_factor is None:
            if self._segment_factor is None:
                self._segment_factor = 1.0
        else:
            if self._segment_factor != segment_factor:
                self._segment_factor = segment_factor
                self._power = None
        if self._window != window:
            self._power = None
            self._window = window
        if self._power is None:
            data = self._get_not_indexed_data()
            self._power = self._compute_power(data)
        if from_freq is None:
            output = self._power
            self._lastCuttedFrequency = self._frequency
        else:
            ul = self._frequency <= to_freq
            dl = self._frequency >= from_freq
            lim = ul & dl
            self._lastCuttedFrequency = self._frequency[lim]
            output = self._power[lim]
        if index is None:
            return output
        return output[:, index]

    def _compute_power(self, data):

        if isinstance(data, NotAvailable):
            raise Exception('Cannot calculate power: data is not available')
        
        delta_time = self.delta_time
        if isinstance(delta_time, u.Quantity):
            value_hz = (1 / delta_time).to_value(u.Hz)
        else:
            value_hz = 1 / delta_time

        self._frequency, x = welch(data.T, value_hz,
                                   window=self._window,
                                   nperseg=data.shape[0] / self._segment_factor)
        df = np.diff(self._frequency)[0]
        return x.T * df

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def time_median(self, times=None, *args, **kwargs):
        '''Median over time for each series'''
        return np.median(self.get_data(*args, times=times, **kwargs), axis=0)

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def time_std(self, times=None, *args, **kwargs):
        '''Standard deviation over time for each series'''
        return np.std(self.get_data(*args, times=times, **kwargs), axis=0)

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def time_average(self, times=None, *args, **kwargs):
        '''Average value over time for each series'''
        return np.mean(self.get_data(*args, times=times, **kwargs), axis=0)

    @modify_help(arg_str='[times=[from,to]], [time_idx]')
    def ensemble_average(self, times=None, *args, **kwargs):
        '''Average across series at each sampling time'''
        return np.mean(self.get_data(*args, times=times, **kwargs), axis=1)

    @modify_help(arg_str='[times=[from,to]], [time_idx]')
    def ensemble_std(self, times=None, *args, **kwargs):
        '''Standard deviation across series at each sampling time'''
        return np.std(self.get_data(*args, times=times, **kwargs), axis=1)

    @modify_help(arg_str='[times=[from,to]], [time_idx]')
    def ensemble_median(self, times=None, *args, **kwargs):
        '''Median across series at each sampling time'''
        return np.median(self.get_data(*args, times=times, **kwargs), axis=1)

    @modify_help(call='plot_hist(from_freq=xx, to_freq=xx, [series_idx])')
    def plot_hist(self, from_t=None, to_t=None,
                  overplot=None, plot_to=None,
                  label=None, *args, **kwargs):
        '''Plot histogram'''
        hist = self.get_data(*args, **kwargs)
        t = self.time()
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
        if self.data_units() is None:
            units = "(a.u.)"
        else:
            units = self.data_units()

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

    @modify_help(call='plot_spectra(from_freq=xx, to_freq=xx, [series_idx])')
    def plot_spectra(self, from_freq=None, to_freq=None,
                     segment_factor=None,
                     overplot=False,
                     label=None, plot_to=None,
                     lineary=False, linearx=False,
                     *args, **kwargs):
        '''Plot PSD'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cutted_frequency()

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

        if self.data_units() is None:
            units = "(a.u.)"
        else:
            units = self.data_units()

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


    @modify_help(call='plot_cumulative_spectra(from_freq=xx, to_freq=xx, [series_idx])')
    def plot_cumulative_spectra(self, from_freq=None, to_freq=None,
                                segment_factor=None,
                                label=None, plot_to=None,
                                overplot=False, plot_rms=False, lineary=False,
                                *args, **kwargs):
        '''Plot cumulative PSD'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cutted_frequency()
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

        if self.data_units() is None:
            units = "(a.u.)"
        else:
            units = self.data_units()

        title = "Cumulated " + label_str
        if self.data_units() is not None:
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


class TimeSeriesWithInterpolation(TimeSeries):
    '''
    :class:`TimeSeries` with automatic interpolation of missing data.

    Missing data points are detected from a jump in the frame counter,
    and are linearly interpolated between valid data points.

    In addition to the methods defined by :class:`TimeSeries`, the derived
    class must also implement a `_get_counter()` method that returns
    the (potentially incomplete) frame counter array. The frame counter
    can be of any integer or floating point type, and can increase by any
    amount at each time step, as long as it is regular, and can start
    from any value.
    These are all valid frame counters (some have gaps in them)::

        [0,1,2,3,4,5,6]
        [-6, -3, 0, 3, 6, 9, 15]
        [1.0, 1.2, 1.4, 2.0, 2.2, 2.4]

    Interpolation is an expensive operation and is not automatic.
    The derived class must call the interpolation routine in the
    `_get_not_indexed_data()` method explicitly.
    The data array passed to `interpolate_missing_data()` must not
    include the missing points: if the "theoretical" shape is (100,n) but
    one frame is missing, the data array must have shape (99,n) and the
    frame counter (99,). The interpolated data and frame counter will
    have the correct dimensions.

    For example::

        def _get_counter(self):
            return fits.getdata('file_with_incomplete_frame_counter.fits')

        def _get_not_indexed_data(self):
            raw_data = fits.getdata('file_with_incomplete_data.fits')
            return self.interpolate_missing_data(raw_data)

    Since interpolation can be slow, it is recommended that some form of
    caching strategy is implemented in the `_get_not_indexed_data()` method.
    '''

    # TODO remove it?
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        TimeSeries.__init__(self)
        self._counter = None

    def get_counter(self):
        '''Returns the interpolated frame counter array'''
        if self._counter is None:
            self._counter = self._interpolate_counter()
        return self._counter

    def _interpolate_counter(self):
        counter = self._get_time_vector()
        if isinstance(counter, NotAvailable):
            return NotAvailable()
        step = np.median(np.diff(counter))
        n = round((max(counter) - min(counter)) / step) + 1
        if n == len(counter):
            return counter
        else:
            return np.arange(n) * step + min(counter)

    def interpolate_missing_data(self, data):
        '''
        Interpolate missing data.

        Parameters
        ----------
        data: ndarray
            the original data

        Returns
        -------
        ndarray
            the interpolated array

        Raises
        ------
        ValueError
            if the frame counter first dimension does not have the same length
            as the data first dimension.
        '''
        counter = self._get_time_vector()
        if isinstance(counter, NotAvailable):
            return NotAvailable()

        if data.shape[0] != counter.shape[0]:
            raise ValueError('Shape mismatch between frame counter and data:'
                              + ' - Data: %s' % str(data.shape)
                              + ' - Counter: %s' % str(counter.shape))

        self._counter = self._interpolate_counter()

        # No interpolation done
        if len(self._counter) == len(self._get_time_vector()):
            return data

        new_data = np.zeros((self._counter.shape[0], data.shape[1]))

        # Normalize original counter to unsigned integer with unitary steps,
        # keeping the gaps. It makes easier to use the counter as
        # slice indexes, which must be integers.
        step = np.median(np.diff(counter))
        mycounter = np.round((counter - min(counter)) / step).astype(np.uint32)
        deltas = np.diff(mycounter)
        jumps = np.where(deltas > 1)[0]

        # Data before the first jump
        new_data[:jumps[0] + 1] = data[:jumps[0] + 1]

        shift = 0
        jump_idx = np.concatenate((jumps, [len(new_data)]))

        for j, nextj in pairwise(jump_idx):
            n_interp = deltas[j]
            gap = n_interp - 1
            interp = np.outer(np.arange(0, n_interp),
                              (data[j + 1] - data[j]) / n_interp) + data[j]

            # Interpolated data
            new_data[shift+j : shift+j+n_interp] = interp

            # Original data up to the next jump
            new_data[shift+j+n_interp: shift+nextj+n_interp] = data[j+1:nextj+1]

            # Keep track of how much data has been inserted in new_data
            shift += gap

        return new_data

# ___oOo___
