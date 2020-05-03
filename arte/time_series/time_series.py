import abc
import numpy as np
import functools
from scipy.signal.spectral import welch
from arte.utils.not_available import NotAvailable
from arte.utils.help import add_help, modify_help
from arte.utils.iterators import pairwise


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

    def __init__(self, sampling_interval):
        self.__delta_time = sampling_interval
        self._data = None
        self._frequency = None
        self._lastCuttedFrequency = None
        self._power = None
        self._segment_factor = None
        self._window = None

    @abc.abstractmethod
    def _get_not_indexed_data(self):
        pass

    def get_data(self, *args, **kwargs):
        '''Raw data as a matrix [time, series]'''

        not_indexed_data = self._get_not_indexed_data()
        index = self.get_index_of(*args, **kwargs)
        if index is None:
            return not_indexed_data
        else:
            return not_indexed_data[:, index]

    @abc.abstractmethod
    def get_index_of(self, *args, **kwargs):
        pass

    @property
    def delta_time(self):
        '''Property with the interval between samples (astropy units)'''
        return self.__delta_time

    @delta_time.setter
    def delta_time(self, time):
        self.__delta_time = time

    def frequency(self):
        return self._frequency

    def last_cutted_frequency(self):
        return self._lastCuttedFrequency

    def ensemble_size(self):
        '''Number of distinct series in this time enseble'''
        not_indexed_data = self._get_not_indexed_data()
        return not_indexed_data.shape[1]

    def _apply(self, func, times=None, *args, **kwargs):
        '''Extract data and apply the passed function'''
        data = self.get_data(*args, **kwargs)
        if times is None:
            result = func(data)
        else:
            idxs = np.array(np.arange(times[0], times[1]) / self.__delta_time,
                            dtype='int32')
            result = func(data[idxs])
        return result

    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def power(self, from_freq=None, to_freq=None,
              segment_factor=None, window='boxcar', *args, **kwargs):
        '''PSD across specified series'''

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
        if isinstance(self.__delta_time, NotAvailable):
            raise Exception('Cannot calculate power: deltaTime is not available')

        if isinstance(data, NotAvailable):
            raise Exception('Cannot calculate power: data is not available')

        self._frequency, x = welch(data.T, (1 / self.__delta_time).value,
                                   window=self._window,
                                   nperseg=data.shape[0] / self._segment_factor)
        df = np.diff(self._frequency)[0]
        return x.T * df

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def time_median(self, times=None, *args, **kwargs):
        '''Median over time for each series'''
        func = functools.partial(np.median, axis=0)
        return self._apply(func, times, *args, **kwargs)

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def time_std(self, times=None, *args, **kwargs):
        '''Standard deviation over time for each series'''
        func = functools.partial(np.std, axis=0)
        return self._apply(func, times, *args, **kwargs)

    @modify_help(arg_str='[times=[from,to]], [series_idx]')
    def time_average(self, times=None, *args, **kwargs):
        '''Average value over time for each series'''
        func = functools.partial(np.mean, axis=0)
        return self._apply(func, times, *args, **kwargs)

    @modify_help(arg_str='[times=[from,to]], [time_idx]')
    def ensemble_average(self, times=None, *args, **kwargs):
        '''Average across series at each sampling time'''
        func = functools.partial(np.mean, axis=1)
        return self._apply(func, times, *args, **kwargs)

    @modify_help(arg_str='[times=[from,to]], [time_idx]')
    def ensemble_std(self, times=None, *args, **kwargs):
        '''Standard deviation across series at each sampling time'''
        func = functools.partial(np.std, axis=1)
        return self._apply(func, times, *args, **kwargs)

    @modify_help(arg_str='[times=[from,to]], [time_idx]')
    def ensemble_median(self, times=None, *args, **kwargs):
        '''Median across series at each sampling time'''
        func = functools.partial(np.median, axis=1)
        return self._apply(func, times, *args, **kwargs)

    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def plot_spectra(self, from_freq=None, to_freq=None,
                     segment_factor=None,
                     overplot=False,
                     label=None,
                     *args, **kwargs):
        '''Plot the PSD across specified series'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cutted_frequency()

        import matplotlib.pyplot as plt
        if not overplot:
            plt.cla()
            plt.clf()
        plt.plot(freq[1:], power[1:], label=label)
        plt.loglog()
        plt.xlabel('f [Hz]')
        plt.ylabel('psd [V^2]')
        if label is not None:
            plt.legend()
        return plt

    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def plot_cumulative_spectra(self, from_freq=None, to_freq=None,
                                segment_factor=None,
                                overplot=False, *args, **kwargs):
        '''Plot the cumulative PSD across specified series'''
        power = self.power(from_freq, to_freq,
                           segment_factor,
                           *args, **kwargs)
        freq = self.last_cutted_frequency()

        import matplotlib.pyplot as plt
        if not overplot:
            plt.cla()
            plt.clf()
        plt.plot(freq[1:], np.cumsum(power, 0)[1:])
        plt.loglog()
        plt.xlabel('f [Hz]')
        plt.ylabel('cumsum(psd) [V^2]')
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

    def __init__(self, sampling_interval):
        TimeSeries.__init__(self, sampling_interval)
        self._counter = None
        self._original_counter = None

    def get_original_counter(self):
        '''Returns the original frame counter array'''
        if self._original_counter is None:
            self._original_counter = self._get_counter()
        return self._original_counter

    def get_counter(self):
        '''Returns the interpolated frame counter array'''
        if self._counter is None:
            self._counter = self._interpolate_counter()
        return self._counter

    @abc.abstractmethod
    def _get_counter(self):
        pass

    def _interpolate_counter(self):
        counter = self.get_original_counter()
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
        counter = self.get_original_counter()
        if isinstance(counter, NotAvailable):
            return NotAvailable()

        if data.shape[0] != counter.shape[0]:
            raise ValueError('Shape mismatch between frame counter and data:'
                              + ' - Data: %s' % str(data.shape)
                              + ' - Counter: %s' % str(counter.shape))

        self._counter = self._interpolate_counter()

        # No interpolation done
        if len(self._counter) == len(self.get_original_counter()):
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
