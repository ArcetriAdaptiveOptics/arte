import abc
import math
from functools import cached_property

import numpy as np
from scipy.signal import welch
from astropy import units as u

from arte.utils.not_available import NotAvailable
from arte.utils.help import add_help, modify_help
from arte.utils.iterators import pairwise
from arte.utils.unit_checker import make_sure_its_a



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
        return self._index_data(data, index)
        
    def _index_data(self, data, index):
        if index is None:
            return data
        elif isinstance(index, tuple):
            return data[(slice(0,data.shape[0]),) + index]
        else:
            return data[:, index]

    @abc.abstractmethod
    def get_index_of(self, *args, **kwargs):
        pass

    def data_unit(self):
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
        return math.prod(not_indexed_data.shape[1:])

    def time_size(self):
        '''Number of time samples in this time ensemble'''
        not_indexed_data = self._get_not_indexed_data()
        return not_indexed_data.shape[0]

    def _data_flattened(self, data):
        '''Flatten data over all non-time dimensions'''
        return data.reshape(data.shape[0], self._data_sample_size(data))
    
    def _data_sample_shape(self, data):
        return data.shape[1:]
    
    def _data_sample_axes(self, data):
        return tuple(range(len(data.shape))[1:])
    
    def _data_sample_size(self, data):
        return math.prod(data.shape[1:])
        
    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def power(self, from_freq=None, to_freq=None,
              segment_factor=None, window='boxcar', *args, **kwargs):
        '''Power Spectral Density across specified series'''

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
            # Perform power on flattened data and restore the shape afterwards
            power = self._compute_power(self._data_flattened(data))
            self._power = power.reshape((power.shape[0],) + self._data_sample_shape(data))
        if from_freq is None:
            output = self._power
            self._lastCuttedFrequency = self._frequency
        else:
            ul = self._frequency <= to_freq
            dl = self._frequency >= from_freq
            lim = ul & dl
            self._lastCuttedFrequency = self._frequency[lim]
            output = self._power[lim]
        index = self.get_index_of(*args, **kwargs)
        return self._index_data(output, index)

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

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def time_median(self, *args, times=None, **kwargs):
        '''Median over time for each series'''
        return np.median(self.get_data(*args, times=times, **kwargs), axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def time_std(self, *args, times=None, **kwargs):
        '''Standard deviation over time for each series'''
        return np.std(self.get_data(*args, times=times, **kwargs), axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def time_average(self, *args, times=None, **kwargs):
        '''Average value over time for each series'''
        return np.mean(self.get_data(*args, times=times, **kwargs), axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_average(self, *args, times=None, **kwargs):
        '''Average across series at each sampling time'''
        data = self.get_data(*args, times=times, **kwargs)
        return np.mean(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_std(self, *args, times=None, **kwargs):
        '''Standard deviation across series at each sampling time'''
        data = self.get_data(*args, times=times, **kwargs)
        return np.std(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_median(self, *args, times=None, **kwargs):
        '''Median across series at each sampling time'''
        data = self.get_data(*args, times=times, **kwargs)
        return np.median(data, axis=self._data_sample_axes(data))

    @modify_help(call='plot_hist([series_idx], from_freq=xx, to_freq=xx, )')
    def plot_hist(self, *args, from_t=None, to_t=None,
                  overplot=None, plot_to=None,
                  label=None,  **kwargs):
        '''Plot histogram. TODO: does not work, rewrite'''
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


# ___oOo___
