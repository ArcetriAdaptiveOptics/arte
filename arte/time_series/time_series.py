import abc
import math
import collections
from functools import cache, cached_property

import numpy as np
from scipy.signal import welch
from astropy import units as u

from arte.time_series.axis_handler import AxisHandler
from arte.utils.not_available import NotAvailable
from arte.utils.help import add_help, modify_help
from arte.utils.unit_checker import make_sure_its_a


class TimeSeriesException(Exception):
    '''Exception raised by TimeSeries'''


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

    def __init__(self, axes=None):
        self._frequency = None
        self._last_cut_frequency = None
        self._power = None
        self._segment_factor = None
        self._window = None
        self._axis_handler = AxisHandler(axes)

    @abc.abstractmethod
    def _get_not_indexed_data(self):
        pass

    @property
    def axes(self):
        '''Data series shape'''
        return self._axis_handler.axes()

    @cache
    def _get_time_vector(self):
        '''Override to provide a custom time vector'''
        return np.arange(len(self._get_not_indexed_data()))
    
    def _get_time_vector_check(self):
        v = self._get_time_vector()
        if isinstance(v, np.ndarray):
            return v
        else:
            try:
                # Try to convert
                return np.array(v)
            except TypeError as e:
                raise TimeSeriesException('Cannot convert _get_time_vector() result to numpy array') from e

    def get_data(self, *args, times=None, axes=None, **kwargs):
        '''Raw data as a matrix [time, series]'''

        data = self._get_not_indexed_data()
        if times is not None:
            if not isinstance(times, collections.abc.Sequence) or len(times) != 2:
                raise TimeSeriesException('Times keywords must be a sequence of two elements: [start, stop]')
            time_vector = self.get_time_vector()
            if len(time_vector) != len(data):
                raise TimeSeriesException('Time vector and data lengths differ')

            start, stop = times
            if isinstance(time_vector, u.Quantity):
                if start is not None:
                    start = make_sure_its_a(time_vector.unit, start)
                if stop is not None:
                    stop = make_sure_its_a(time_vector.unit, stop)

            idxs = np.ones(len(time_vector), dtype=bool)
            if start is not None:
                idxs = np.logical_and(idxs, time_vector >= start)
            if stop is not None:
                idxs = np.logical_and(idxs, time_vector < stop)
            data = data[idxs]

        index = self.get_index_of(*args, **kwargs)
        data = self._index_data(data, index)
        data = self._axis_handler.transpose(data, axes)
        return data

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
        '''Property with the interval between samples.
        
        If no interval can be determined (time vector too short),
        returns 1 with the correct unit if applicable.
        '''
        time_vector = self.get_time_vector()
        if len(time_vector) == 0:
            return 1
        diff = np.diff(time_vector)
        if len(diff) > 0:
            return np.median(diff)
        else:
            if isinstance(time_vector[0], u.Quantity):
                return 1 * time_vector[0].unit
            else:
                return 1

    def frequency(self):
        return self._frequency

    def get_time_vector(self):
        '''Return the series time vector'''
        return self._get_time_vector_check()

    def last_cut_frequency(self):
        return self._last_cut_frequency

    @cache
    def ensemble_size(self):
        '''Number of distinct series in this time ensemble'''
        not_indexed_data = self._get_not_indexed_data()
        return math.prod(not_indexed_data.shape[1:])

    @cache
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
    def time_rms(self, *args, times=None, **kwargs):
        '''Root-Mean-Square value over time for each series'''
        x = self.get_data(*args, times=times, **kwargs)
        return np.sqrt(np.mean(np.abs(x)**2, axis=0))


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

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_rms(self, *args, times=None, **kwargs):
        '''Root-Mean-Square across series at each sampling time'''
        x = self.get_data(*args, times=times, **kwargs)
        return np.sqrt(np.mean(np.abs(x)**2, axis=1))

    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def power(self, *args, from_freq=None, to_freq=None,
              segment_factor=None, window='boxcar', **kwargs):
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
            self._last_cut_frequency = self._frequency
        else:
            ul = self._frequency <= to_freq
            dl = self._frequency >= from_freq
            ul |= np.isclose(self._frequency, to_freq)
            dl |= np.isclose(self._frequency, from_freq)
            lim = ul & dl
            self._last_cut_frequency = self._frequency[lim]
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


# ___oOo___
