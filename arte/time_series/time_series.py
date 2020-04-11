import abc
import numpy as np
from astropy.io import fits
from scipy.signal.spectral import welch
from arte.utils.not_available import NotAvailable
from arte.utils.help import ThisClassCanHelp, add_to_help


class TimeSeries(ThisClassCanHelp, metaclass=abc.ABCMeta):
    '''
    Base class implementing operations on data representing time series.

    The derived class must implement a get_data() method that returns a 
    numpy array of shape (n_time_elements, n_ensemble_elements)

    The derived class can implement a get_index_of() method to add 
    ensemble-indexing capabilities (e.g. returning of partial subset
    based on names).

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

    @add_to_help(arg_str='[time_idx]')
    def ensemble_average(self, *args, **kwargs):
        ''' Average across series at each sampling time '''
        data = self.get_data(*args, **kwargs)
        return np.mean(data, axis=1)

    @add_to_help(arg_str='[time_idx]')
    def ensemble_std(self, *args, **kwargs):
        ''' Standard deviation across series at each sampling time '''
        data = self.get_data(*args, **kwargs)
        return np.std(data, axis=1)

    @add_to_help(arg_str='[series_idx]')
    def time_average(self, times=None, *args, **kwargs):
        '''Average value over time for each series'''
        data = self.get_data(*args, **kwargs)
        if times is None:
            timeAverage = np.mean(data, axis=0)
        else:
            idxs = np.array(np.arange(times[0], times[1]) / self.__deltaTime,
                            dtype='int32')
            timeAverage = np.mean(data[idxs], axis=0)
        return timeAverage

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

    @add_to_help(arg_str='[time_idx]')
    def time_median(self, times=None, *args, **kwargs):
        '''Median over time for each series'''
        data = self.get_data(*args, **kwargs)
        if times is None:
            timeMedian = np.median(data, axis=0)
        else:
            idxs = np.array(np.arange(times[0], times[1]) / self.__deltaTime,
                            dtype='int32')
            timeMedian = np.median(data[idxs], axis=0)
        return timeMedian

    @add_to_help(doc_str='Standard deviation in time for each series', arg_str='series_idx')
    def time_std(self, times=None, *args, **kwargs):
        data = self.get_data(*args, **kwargs)
        if times is None:
            timeStd = np.std(data, axis=0)
        else:
            idxs = np.array(np.arange(times[0], times[1]) / self.__deltaTime,
                            dtype='int32')
            timeStd = np.std(data[idxs], axis=0)
        return timeStd

    def plot_spectra(self, from_freq=None, to_freq=None,
                     segment_factor=None,
                     overplot=False,
                     label=None,
                     *args, **kwargs):
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

    def plot_cumulative_spectra(self, from_freq=None, to_freq=None,
                                segment_factor=None,
                                overplot=False, *args, **kwargs):
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

    def _get_counter(self):
        if self._counter is None:
            filename = self._file_name_walker.LGSWCCDFrameCounters()
            self._counter = fits.getdata(filename)
        return self._counter


class TimeSeriesWithInterpolation(TimeSeries):

    # TODO remove it?
    __metaclass__ = abc.ABCMeta

    def __init__(self, samplingInterval):
        TimeSeries.__init__(self, samplingInterval)
        self.__deltaTime = samplingInterval
        self._newCounter = None
        self._file_name_walker = None

    def _get_counter(self):
        if self._counter is None:
            filename = self._file_name_walker.slopesFrameCounters()
            counter = fits.getdata(filename)
            if counter.size is 0:
                self._counter = NotAvailable()
            else:
                if counter[0] > counter[-1]:
                    counter[np.argmax(counter) + 1:] += np.max(counter)
                self._counter = counter
        return self._counter

    def _get_new_counter(self):
        if self._headerParser.isAoLoopStatusClosed():
            if self._newCounter is None:
                self._calculate_new_counter()
            return np.array(self._newCounter, dtype=int)
        else:
            return self._get_counter()

    def _calculate_new_counter(self):
        counter = self._get_counter()
        if isinstance(counter, NotAvailable):
            self._newCounter = NotAvailable()
        step = np.median(np.diff(counter))
        self._newCounter = np.arange(counter[0], counter[-1] + step,
                                     step, dtype=np.int)

    def interpolate_missing_data(self, data):
        counter = self._get_counter()
        if data.shape[0] == counter.shape[0]:
            self._calculate_new_counter()
            newData = np.zeros((self._newCounter.shape[0], data.shape[1]))
            dc = np.diff(counter)
            step = np.median(dc)
            jumps = np.where(dc > step)[0]
            nc = 0
            j1 = 0
            for j in jumps:
                newData[nc + j1:nc + j + step] = data[j1:j + step]
                newData[nc + j + step:nc + j + dc[j]] = np.outer(np.arange(step, dc[j]),
                                                                 (data[j + step] - data[j]) / dc[j]) + data[j]
                nc += (dc[j] - step) / step
                j1 = j + step
            newData[nc + j1:] = data[j1:]
        else:
            newData = NotAvailable()
        return newData
