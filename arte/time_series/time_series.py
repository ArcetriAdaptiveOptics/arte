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
    Base class for time series analysis of adaptive optics telemetry data.

    This class provides a framework for analyzing time-evolving data where multiple
    "sister" quantities (ensemble elements) are sampled simultaneously over time.
    Examples include:
    
    - Camera pixel intensities evolving over multiple frames
    - Deformable mirror actuator commands over time
    - Wavefront sensor slopes for multiple subapertures
    - Modal coefficients (Zernike, KL modes) during AO operation
    
    The data structure is always `(n_time_samples, ...ensemble_shape...)` where
    the first dimension is time and remaining dimensions define the ensemble geometry.
    
    Subclasses must implement:
    
    - :meth:`_get_not_indexed_data`: Return raw data as numpy array with time as first axis
    - :meth:`get_index_of`: Define ensemble indexing logic (select specific elements)
    
    The class supports:
    
    - Time-domain statistics (mean, std, rms over time for each ensemble element)
    - Ensemble statistics (mean, std, rms across ensemble at each time step)
    - Power spectral density analysis with Welch method
    - Time-based filtering (analyze specific time intervals)
    - Physical units support via astropy.units
    - Masked arrays for missing/invalid data (e.g., points outside circular pupil)
    
    Parameters
    ----------
    axes : sequence of str, optional
        Named axes for the ensemble dimensions, enabling axis transposition.
        If None, no axis reordering is possible.
    
    Notes
    -----
    MaskedArray support: Use `numpy.ma.MaskedArray` to handle missing data in the
    ensemble dimension (e.g., wavefront defined only inside a circular pupil).
    Temporal masking is possible but may cause issues with FFT-based operations
    if sampling becomes non-uniform.
    
    Originally implemented as part of the ARGOS codebase.
    
    See Also
    --------
    MultiTimeSeries : Combine multiple TimeSeries with different sampling rates
    
    Examples
    --------
    >>> class MyTimeSeries(TimeSeries):
    ...     def __init__(self, data):
    ...         super().__init__()
    ...         self._data = data  # shape: (n_times, n_elements)
    ...     
    ...     def _get_not_indexed_data(self):
    ...         return self._data
    ...     
    ...     def get_index_of(self, *args, **kwargs):
    ...         if len(args) == 0:
    ...             return None  # Return all elements
    ...         return args[0]   # Return specific index
    >>> 
    >>> # Analyze DM commands for 1000 time steps, 100 actuators
    >>> dm_commands = np.random.randn(1000, 100)
    >>> ts = MyTimeSeries(dm_commands)
    >>> rms_per_actuator = ts.time_rms()  # RMS over time for each actuator
    >>> avg_command = ts.ensemble_average()  # Average across actuators at each time
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
        '''
        Retrieve time series data with optional filtering and indexing.
        
        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`get_index_of` for ensemble element selection.
        times : sequence of 2 elements, optional
            Time range `[start, stop]` for filtering. Can be None (no filtering),
            or contain None elements for one-sided filtering.
            Units must match the time vector if using astropy quantities.
        axes : sequence of str, optional
            Reorder axes to specified order. Requires axis names to be defined
            during initialization.
        
        Returns
        -------
        ndarray
            Data array with shape `(n_times, ...ensemble_shape...)` after
            applying time filtering, ensemble indexing, and axis transposition.
        
        Examples
        --------
        >>> # Get all data
        >>> data = ts.get_data()
        >>> 
        >>> # Get data for specific time range (closed-loop only)
        >>> data = ts.get_data(times=[1.0 * u.s, 5.0 * u.s])
        >>> 
        >>> # Get data for specific ensemble elements
        >>> data = ts.get_data(modes=[2, 3, 4])  # Assuming mode-based indexing
        '''

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
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.median(data, axis=0)
        return np.median(data, axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def time_std(self, *args, times=None, **kwargs):
        '''Standard deviation over time for each series'''
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.std(data, axis=0)
        return np.std(data, axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def time_average(self, *args, times=None, **kwargs):
        '''Average value over time for each series'''
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.mean(data, axis=0)
        return np.mean(data, axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def time_rms(self, *args, times=None, **kwargs):
        '''Root-Mean-Square value over time for each series'''
        x = self.get_data(*args, times=times, **kwargs)
        if isinstance(x, np.ma.MaskedArray):
            return np.sqrt(np.ma.mean(np.abs(x)**2, axis=0))
        return np.sqrt(np.mean(np.abs(x)**2, axis=0))


    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_average(self, *args, times=None, **kwargs):
        '''Average across series at each sampling time'''
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.mean(data, axis=self._data_sample_axes(data))
        return np.mean(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_std(self, *args, times=None, **kwargs):
        '''Standard deviation across series at each sampling time'''
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.std(data, axis=self._data_sample_axes(data))
        return np.std(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_median(self, *args, times=None, **kwargs):
        '''Median across series at each sampling time'''
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.median(data, axis=self._data_sample_axes(data))
        return np.median(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def ensemble_rms(self, *args, times=None, **kwargs):
        '''Root-Mean-Square across series at each sampling time'''
        x = self.get_data(*args, times=times, **kwargs)
        if isinstance(x, np.ma.MaskedArray):
            return np.sqrt(np.ma.mean(np.abs(x)**2, axis=1))
        return np.sqrt(np.mean(np.abs(x)**2, axis=1))

    @modify_help(call='power(from_freq=xx, to_freq=xx, [series_idx])')
    def power(self, *args, from_freq=None, to_freq=None,
              segment_factor=None, window='boxcar', **kwargs):
        '''
        Compute Power Spectral Density using Welch's method.
        
        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`get_index_of` for ensemble element selection.
        from_freq : float, optional
            Lower frequency bound for output. If None, starts from DC.
        to_freq : float, optional
            Upper frequency bound for output. If None, goes to Nyquist.
        segment_factor : float, optional
            Segment length factor for Welch method. Larger values give better
            frequency resolution but worse variance. Default is 1.0 (one segment).
        window : str, optional
            Window function name for Welch method. Default is 'boxcar' (no windowing).
            See scipy.signal.welch for available windows.
        
        Returns
        -------
        ndarray
            Power spectral density with shape `(n_frequencies, ...ensemble_shape...)`.
            Units are data_unit^2 * Hz if applicable.
        
        Notes
        -----
        The PSD is computed using scipy.signal.welch and normalized by frequency
        bin width. Results are cached unless segment_factor or window changes.
        
        For MaskedArrays, masked values are filled with zeros before FFT computation.
        
        Examples
        --------
        >>> # Compute full PSD
        >>> psd = ts.power()
        >>> freq = ts.frequency()
        >>> 
        >>> # Analyze only low frequencies (< 100 Hz)
        >>> psd = ts.power(from_freq=0, to_freq=100)
        >>> 
        >>> # Use Hanning window with 4 segments for variance reduction
        >>> psd = ts.power(segment_factor=4, window='hann')
        '''

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
        
        # Convert MaskedArray to regular array for scipy.signal.welch compatibility
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(0)
        
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
