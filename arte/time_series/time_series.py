import abc
import numpy as np
import functools
from astropy.io import fits
from scipy.signal.spectral import welch
from arte.utils.not_available import NotAvailable
from arte.utils.help import ThisClassCanHelp, add_to_help


class TimeSeries(ThisClassCanHelp, metaclass=abc.ABCMeta):
    '''
    Base class implementing operations on data representing time series.

    The derived class must implement a `_get_not_indexed_data()` method
    that returns a numpy array of shape (n_time_elements, n_ensemble_elements).

    The derived class must also implement a `get_index_of()` method to add 
    ensemble indexing with arbitrary \*args and \*\*kwargs parameters
    (e.g. returning of partial subset based on indxes or names).

    Originally implemented as part of the ARGOS codebase.
    '''

    def __init__(self, samplingInterval):
        self.__deltaTime = samplingInterval
        self._data = None
        self._frequency = None
        self._lastCuttedFrequency = None
        self._timeAverage = None
        self._timeStd = None
        self._power = None
        self._prefix = None
        self._segment_factor = None
        self._window = None
        self._counter = None

    @abc.abstractmethod
    def _get_not_indexed_data(self):
        pass

    @add_to_help
    def get_data(self, *args, **kwargs):
        '''
        Raw data as a matrix [time, series]
        '''
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
    @add_to_help
    def delta_time(self):
        '''Time interval between samples (astropy units)'''
        return self.__deltaTime

    @delta_time.setter
    def delta_time(self, time):
        self.__deltaTime = time

    def frequency(self):
        return self._frequency

    def last_cutted_frequency(self):
        return self._lastCuttedFrequency
 
    @add_to_help
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
            idxs = np.array(np.arange(times[0], times[1]) / self.__deltaTime,
                            dtype='int32')
            result = func(data[idxs])
        return result
        
    @add_to_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
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
        if isinstance(self.__deltaTime, NotAvailable):
            raise Exception('Cannot calculate power: deltaTime is not available')

        if isinstance(data, NotAvailable):
            raise Exception('Cannot calculate power: data is not available')

        self._frequency, x = welch(data.T, (1 / self.__deltaTime).value,
                                   window=self._window,
                                   nperseg=data.shape[0] / self._segment_factor)
        df = np.diff(self._frequency)[0]
        return x.T * df

    @add_to_help(arg_str='[series_idx]')
    def time_median(self, times=None, *args, **kwargs):
        '''Median over time for each series'''
        func = functools.partial(np.median, axis=0)
        return self._apply(func, times, *args, **kwargs)
    
    @add_to_help(doc_str='[series_idx]')
    def time_std(self, times=None, *args, **kwargs):
        '''Standard deviation over time for each series'''
        func = functools.partial(np.std, axis=0)
        return self._apply(func, times, *args, **kwargs)

    @add_to_help(arg_str='[series_idx]')
    def time_average(self, times=None, *args, **kwargs):
        '''Average value over time for each series'''
        func = functools.partial(np.mean, axis=0)
        return self._apply(func, times, *args, **kwargs)

    @add_to_help(arg_str='[time_idx]')
    def ensemble_average(self, times=None, *args, **kwargs):
        '''Average across series at each sampling time '''
        func = functools.partial(np.mean, axis=1)
        return self._apply(func, times, *args, **kwargs)

    @add_to_help(arg_str='[time_idx]')
    def ensemble_std(self, times=None, *args, **kwargs):
        '''Standard deviation across series at each sampling time '''
        func = functools.partial(np.std, axis=1)
        return self._apply(func, times, *args, **kwargs)

    @add_to_help(arg_str='[time_idx]')
    def ensemble_median(self, times=None, *args, **kwargs):
        '''Median across series at each sampling time '''
        func = functools.partial(np.median, axis=1)
        return self._apply(func, times, *args, **kwargs)

    @add_to_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
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

    @add_to_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
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
    the (potentially incomplete) frame counter array.

    Interpolated is not automatic. The derived class must call
    this routine in the `_get_not_indexed_data()` method explicitly.
    For example::
        
        def _get_counter(self):
            return fits.getdata('frame_counter.fits')
            
        def _getNotIndexedData(self, *args, **kwargs):
            raw_data = fits.getdata('file_with_incomplete_data.fits')
            return self.interpolate_missing_data(raw_data)
        
    '''
    # TODO remove it?
    __metaclass__ = abc.ABCMeta

    def __init__(self, samplingInterval):
        TimeSeries.__init__(self, samplingInterval)
        self.__deltaTime = samplingInterval
        self._counter = None
        self._original_counter = None
 
    @add_to_help
    def get_original_counter(self):
        '''Returns the original frame counter array'''
        if self._original_counter is None:
            self._original_counter = self._get_counter()
        return self._original_counter

    @add_to_help
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
        return np.arange(counter[0], counter[-1] + step, step, dtype=np.int)

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
        '''
        counter = self.get_original_counter()
        if data.shape[0] == counter.shape[0]:
            self._counter = self._interpolate_counter()
            new_data = np.zeros((self._counter.shape[0], data.shape[1]))
            dc = np.diff(counter)
            step = int(np.median(dc))
            jumps = np.where(dc > step)[0]
            nc = 0
            j1 = 0
            for j in jumps:
                new_data[nc + j1:nc + j + step] = data[j1:j + step]
                interp = np.outer(np.arange(step, dc[j]),
                                  (data[j + step] - data[j]) / dc[j]) + data[j]
                new_data[nc + j + step:nc + j + dc[j]] = interp
                                                               
                nc += (dc[j] - step) // step
                j1 = j + step
            new_data[nc + j1:] = data[j1:]
        else:
            new_data = NotAvailable()
        return new_data
