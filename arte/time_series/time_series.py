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
    
    The class provides a **chainable fluent API** for data analysis:
    
    - Ensemble reductions: `ensemble_rms`, `ensemble_mean`, `ensemble_std`, `ensemble_median`, `ensemble_ptp`
    - Time reductions: `time_mean`, `time_std`, `time_rms`, `time_median`, `time_ptp`
    - Filtering: `filter(modes=..., times=...)` or `with_times(...)`
    - Value extraction: `.value` property (like pandas)
    - Chain operations: `ts.filter(modes=[2,3]).ensemble_rms.time_mean.value`
    
    Additional features:
    
    - Power spectral density analysis with Welch method
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
    >>> 
    >>> # Fluent API (chainable)
    >>> mean_rms = ts.ensemble_rms.time_mean.value  # Scalar
    >>> rms_time_series = ts.filter(elements=[0,1,2]).ensemble_rms.value  # Array
    >>> long_exposure_ptp = ts.time_mean.ensemble_ptp.value  # Peak-to-valley
    '''

    def __init__(self, axes=None):
        self._frequency = None
        self._last_cut_frequency = None
        self._power = None
        self._segment_factor = None
        self._window = None
        self._axis_handler = AxisHandler(axes)

    def __array__(self, dtype=None, copy=None):
        """
        Numpy compatibility protocol (numpy 2.0+ compatible).
        
        Enables using TimeSeries directly in numpy operations and matplotlib:
        - plt.plot(time, ts) - automatic conversion
        - np.mean(ts) - works directly
        
        Parameters
        ----------
        dtype : numpy dtype, optional
            Desired dtype for the array (numpy 2.0+)
        copy : bool, optional
            Whether to copy the data (numpy 2.0+)
        
        Returns
        -------
        ndarray
            Underlying data array
        
        Examples
        --------
        >>> ts = MyTimeSeries(data)
        >>> rms = ts.ensemble_rms_property  # Returns TimeSeries
        >>> plt.plot(rms)  # Works! __array__() called automatically
        >>> mean_rms = np.mean(rms)  # Works!
        """
        arr = self._get_not_indexed_data()
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Support numpy universal functions on TimeSeries.
        
        Enables operations like: np.sin(ts), np.sqrt(ts), ts * 2
        
        Parameters
        ----------
        ufunc : numpy ufunc
            The universal function being applied
        method : str
            The ufunc method ('__call__', 'reduce', etc.)
        *inputs : tuple
            Input arrays/objects
        **kwargs : dict
            Additional ufunc arguments
        
        Returns
        -------
        ndarray or scalar
            Result of the ufunc operation
        """
        # Convert TimeSeries inputs to arrays
        arrays = []
        for inp in inputs:
            if isinstance(inp, TimeSeries):
                arrays.append(inp.__array__())
            else:
                arrays.append(inp)
        
        # Apply ufunc and return result (not wrapped in TimeSeries)
        return getattr(ufunc, method)(*arrays, **kwargs)

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
    def get_time_median(self, *args, times=None, **kwargs):
        '''Median over time for each series
        
        .. deprecated::
           Use the `time_median` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_time_median() is deprecated. Use the 'time_median' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.median(data, axis=0)
        return np.median(data, axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_time_std(self, *args, times=None, **kwargs):
        '''Standard deviation over time for each series
        
        .. deprecated:: 
           Use the `time_std` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_time_std() is deprecated. Use the 'time_std' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.std(data, axis=0)
        return np.std(data, axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_time_average(self, *args, times=None, **kwargs):
        '''Average value over time for each series
        
        .. deprecated:: 
           Use the `time_mean` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_time_average() is deprecated. Use the 'time_mean' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.mean(data, axis=0)
        return np.mean(data, axis=0)

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_time_rms(self, *args, times=None, **kwargs):
        '''Root-Mean-Square value over time for each series
        
        .. deprecated::
           Use the `time_rms` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_time_rms() is deprecated. Use the 'time_rms' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        x = self.get_data(*args, times=times, **kwargs)
        if isinstance(x, np.ma.MaskedArray):
            return np.sqrt(np.ma.mean(np.abs(x)**2, axis=0))
        return np.sqrt(np.mean(np.abs(x)**2, axis=0))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_time_ptp(self, *args, times=None, **kwargs):
        '''Peak-to-peak (max - min) over time for each series
        
        .. deprecated::
           Use the `time_ptp` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_time_ptp() is deprecated. Use the 'time_ptp' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.ptp(data, axis=0)
        return np.ptp(data, axis=0)


    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_ensemble_average(self, *args, times=None, **kwargs):
        '''Average across series at each sampling time
        
        .. deprecated::
           Use the `ensemble_mean` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_ensemble_average() is deprecated. Use the 'ensemble_mean' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.mean(data, axis=self._data_sample_axes(data))
        return np.mean(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_ensemble_std(self, *args, times=None, **kwargs):
        '''Standard deviation across series at each sampling time
        
        .. deprecated::
           Use the `ensemble_std` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_ensemble_std() is deprecated. Use the 'ensemble_std' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.std(data, axis=self._data_sample_axes(data))
        return np.std(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_ensemble_median(self, *args, times=None, **kwargs):
        '''Median across series at each sampling time
        
        .. deprecated::
           Use the `ensemble_median` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_ensemble_median() is deprecated. Use the 'ensemble_median' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.median(data, axis=self._data_sample_axes(data))
        return np.median(data, axis=self._data_sample_axes(data))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_ensemble_rms(self, *args, times=None, **kwargs):
        '''
        Root-Mean-Square across series at each sampling time (legacy method).
        
        .. deprecated:: 
            Use the chainable property :attr:`ensemble_rms` for fluent API:
            ``ts.ensemble_rms`` or ``ts.filter(modes=[2,3,4]).ensemble_rms``
            This method will be removed in a future version.
        
        Returns
        -------
        ndarray
            RMS values (not chainable)
        '''
        import warnings
        warnings.warn(
            "get_ensemble_rms() is deprecated. Use the chainable property 'ensemble_rms' instead: "
            "ts.ensemble_rms or ts.filter(modes=[2,3,4]).ensemble_rms",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.sqrt(np.ma.mean(np.abs(data)**2, axis=self._data_sample_axes(data)))
        return np.sqrt(np.mean(np.abs(data)**2, axis=self._data_sample_axes(data)))

    @modify_help(arg_str='[series_idx], [times=[from,to]]')
    def get_ensemble_ptp(self, *args, times=None, **kwargs):
        '''Peak-to-peak (max - min) across series at each sampling time
        
        .. deprecated::
           Use the `ensemble_ptp` property instead. This method will be removed in a future version.
        '''
        import warnings
        warnings.warn(
            "get_ensemble_ptp() is deprecated. Use the 'ensemble_ptp' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.get_data(*args, times=times, **kwargs)
        if isinstance(data, np.ma.MaskedArray):
            return np.ma.ptp(data, axis=self._data_sample_axes(data))
        return np.ptp(data, axis=self._data_sample_axes(data))

    # --- Chainable (Fluent) API ---
    # Upstream filtering methods for fluent API
    
    def filter(self, *args, **kwargs):
        """
        Create filtered TimeSeries by applying ensemble and/or time selection.
        
        Accepts same arguments as get_data(), creating a new TimeSeries
        with pre-filtered data for fluent chaining. Works with any Indexer
        implementation (ModeIndexer, RowColIndexer, DefaultIndexer, custom).
        
        Parameters
        ----------
        *args, **kwargs
            Passed to get_index_of() for ensemble filtering.
            Common kwargs depend on the Indexer:
            - modes= : for ModeIndexer (mode numbers)
            - elements= : for DefaultIndexer (element indices)
            - rows=, cols= : for RowColIndexer (spatial coordinates)
            - coord=, axis= : for interleaved/sequential xy layouts
        times : sequence of 2 elements, optional
            Time range [start, stop] for temporal filtering.
            Can contain None for one-sided filtering.
        
        Returns
        -------
        TimeSeries
            New TimeSeries with filtered data (chainable)
        
        Examples
        --------
        >>> # Modal selection (uses ModeIndexer in derived classes)
        >>> ts.filter(modes=[2, 3, 4]).ensemble_rms.time_mean
        >>> 
        >>> # Time + modal selection
        >>> ts.filter(modes=[2, 3, 4], times=[0.5, 1.0]).ensemble_rms.time_mean
        >>> 
        >>> # Element selection (DefaultIndexer)
        >>> ts.filter(elements=[0, 2, 4, 6]).time_std
        >>> 
        >>> # Row/col selection (RowColIndexer for 2D data)
        >>> ts.filter(rows=slice(0, 10), cols=[5, 10]).ensemble_rms
        >>> 
        >>> # Chainable (filters are accumulated)
        >>> ts.filter(modes=[2, 3, 4]).filter(times=[0.5, 1.0]).ensemble_rms.time_mean
        >>> 
        >>> # Custom indexer kwargs work automatically
        >>> ts.filter(coord='x').time_mean  # for interleaved xy data
        """
        return self._create_filtered_series(*args, **kwargs)
    
    def with_times(self, times):
        """
        Filter to specific time interval (convenience alias for filter(times=...)).
        
        Returns a new chainable TimeSeries containing only data within
        the specified time range [start, stop].
        
        Note: This is an alias for filter(times=...). For filtering by both
        time and ensemble elements, use filter() directly.
        
        Parameters
        ----------
        times : sequence of 2 elements
            Time range [start, stop] (can contain None for one-sided filtering)
        
        Returns
        -------
        TimeSeries
            Time-filtered TimeSeries (chainable)
        
        Examples
        --------
        >>> # Filter time range and compute statistics
        >>> ts.with_times([1, 2]).ensemble_rms.time_mean
        >>> 
        >>> # Chain with filter() for ensemble selection
        >>> ts.filter(modes=[2, 3, 4]).with_times([1, 2]).ensemble_rms
        >>> 
        >>> # Equivalent using filter() directly (preferred)
        >>> ts.filter(modes=[2, 3, 4], times=[1, 2]).ensemble_rms
        """
        return self.filter(times=times)
    
    def _create_filtered_series(self, *args, **kwargs):
        """
        Create filtered view of this TimeSeries.
        
        Returns a new TimeSeries that lazily applies filtering when data
        is accessed. The filtered series supports further chaining.
        
        Parameters
        ----------
        *args
            Positional arguments passed to get_index_of()
        **kwargs
            Keyword arguments for filtering:
            - times : time range [start, stop]
            - Other kwargs passed to get_index_of() (modes=, elements=, rows=, cols=, etc.)
        
        Returns
        -------
        TimeSeries
            Filtered TimeSeries (chainable)
        """
        parent_ts = self
        
        # Extract times separately (handled specially by get_data)
        filter_times = kwargs.pop('times', None)
        filter_args = args
        filter_kwargs = kwargs
        
        class FilteredTimeSeries(TimeSeries):
            def __init__(self):
                super().__init__(axes=parent_ts.axes)
                self._parent = parent_ts
                self._filter_args = filter_args
                self._filter_kwargs = filter_kwargs
                self._filter_times = filter_times
            
            def _get_not_indexed_data(self):
                # Apply parent's filtering: delegate all to get_data()
                return self._parent.get_data(
                    *self._filter_args,
                    times=self._filter_times,
                    **self._filter_kwargs
                )
            
            def get_index_of(self, *args, **kwargs):
                # Delegate to parent's indexing
                return self._parent.get_index_of(*args, **kwargs)
            
            def _create_filtered_series(self, *args, **kwargs):
                # Support chaining: accumulate filters
                # Merge new filters with existing ones
                new_times = kwargs.pop('times', None)
                if new_times is None:
                    new_times = self._filter_times
                
                # Merge kwargs (new ones override existing)
                merged_kwargs = {**self._filter_kwargs, **kwargs}
                
                # For args, we can't easily merge, so new args override
                merged_args = args if args else self._filter_args
                
                return self._parent._create_filtered_series(
                    *merged_args,
                    times=new_times,
                    **merged_kwargs
                )
            
            @cache
            def _get_time_vector(self):
                # Time vector reflects filtering
                if self._filter_times is not None:
                    parent_time = self._parent.get_time_vector()
                    parent_data = self._parent._get_not_indexed_data()
                    
                    # Apply time filtering logic
                    start, stop = self._filter_times
                    idxs = np.ones(len(parent_time), dtype=bool)
                    if start is not None:
                        idxs = np.logical_and(idxs, parent_time >= start)
                    if stop is not None:
                        idxs = np.logical_and(idxs, parent_time < stop)
                    
                    return parent_time[idxs]
                else:
                    return self._parent.get_time_vector()
            
            @cached_property
            def delta_time(self):
                return self._parent.delta_time
        
        return FilteredTimeSeries()
    
    # Chainable operations (properties return TimeSeries)
    
    def _create_reduced_series(self, data, operation_name='reduced'):
        """
        Create new TimeSeries after reduction operation.
        
        Helper for chainable operations that reduce dimensions.
        Preserves time-related metadata in the new series.
        
        Parameters
        ----------
        data : ndarray
            Reduced data (e.g., 1D after ensemble aggregation)
        operation_name : str, optional
            Name of operation for tracking/debugging
        
        Returns
        -------
        TimeSeries
            New TimeSeries instance wrapping the reduced data (chainable)
        
        Notes
        -----
        This creates a minimal TimeSeries subclass that wraps the reduced array.
        Time axis is preserved (axis 0), allowing further temporal operations.
        """
        # Create anonymous subclass with the reduced data
        parent_ts = self
        
        class ReducedTimeSeries(TimeSeries):
            def __init__(self):
                super().__init__()
                self._reduced_data = data
                self._parent = parent_ts
                self._operation = operation_name
            
            def _get_not_indexed_data(self):
                return self._reduced_data
            
            def get_index_of(self, *args, **kwargs):
                # Reduced series has no ensemble indexing
                return None
            
            def __getitem__(self, key):
                """Support numpy-style indexing"""
                return self._reduced_data[key]
            
            @cache
            def _get_time_vector(self):
                # Inherit time vector from parent
                return self._parent.get_time_vector()
            
            @cached_property
            def delta_time(self):
                # Inherit delta_time from parent
                return self._parent.delta_time
        
        return ReducedTimeSeries()
    
    def _create_temporal_reduced_series(self, data, operation_name='time_reduced'):
        """
        Create TimeSeries after temporal reduction operation.
        
        Used for operations that collapse the time dimension (e.g., time_mean, time_std).
        Returns a TimeSeries with time_size=1, using mean time as the single timestamp.
        
        Parameters
        ----------
        data : ndarray
            Temporally reduced data (no time axis, or time_size=1)
        operation_name : str, optional
            Name of operation for tracking/debugging
        
        Returns
        -------
        TimeSeries
            New TimeSeries with shape (1, ...ensemble_shape...) (chainable)
        
        Notes
        -----
        The time_vector for the reduced series contains a single element:
        the mean time of the original series.
        """
        parent_ts = self
        
        # Ensure data has time axis with size 1
        if data.ndim == 0:
            # Scalar → (1,) - use atleast_1d to preserve Quantity
            reduced_data = np.atleast_1d(data)
        elif data.shape[0] != 1:
            # Add time axis: (ensemble,) → (1, ensemble...)
            reduced_data = np.expand_dims(data, axis=0)
        else:
            reduced_data = data
        
        class TemporalReducedTimeSeries(TimeSeries):
            def __init__(self):
                super().__init__(axes=parent_ts.axes)
                self._reduced_data = reduced_data
                self._parent = parent_ts
                self._operation = operation_name
            
            def _get_not_indexed_data(self):
                return self._reduced_data
            
            def get_index_of(self, *args, **kwargs):
                # Delegate to parent for ensemble indexing
                return self._parent.get_index_of(*args, **kwargs)
            
            def __getitem__(self, key):
                """Support numpy-style indexing"""
                return self._reduced_data[key]
            
            @cache
            def _get_time_vector(self):
                # Single-element time vector: mean of parent times
                parent_time = self._parent.get_time_vector()
                if isinstance(parent_time, u.Quantity):
                    return np.array([np.mean(parent_time.value)]) * parent_time.unit
                else:
                    return np.array([np.mean(parent_time)])
            
            @cached_property
            def delta_time(self):
                # Delta time not meaningful for single-timestamp series
                return 0 if not isinstance(self._parent.delta_time, u.Quantity) else 0 * self._parent.delta_time.unit
        
        return TemporalReducedTimeSeries()
    
    @property
    def shape(self):
        """
        Shape of the underlying data array.
        
        Returns the shape tuple of the TimeSeries data, similar to numpy arrays.
        This provides convenient access without requiring explicit conversion.
        
        Returns
        -------
        tuple of int
            Shape of the data array (time, ensemble...)
        
        Examples
        --------
        >>> ts.shape
        (1000, 10)  # 1000 time steps, 10 modes
        >>> ts.ensemble_rms.shape
        (1000,)  # Collapsed to time dimension only
        """
        return self._get_not_indexed_data().shape
    
    @property
    def value(self):
        """
        Extract final value(s) from TimeSeries (convenience accessor).
        
        Returns the underlying data as a numpy array or scalar,
        similar to pandas .values. Useful for getting final results
        from chained operations without explicit numpy conversion.
        
        Returns
        -------
        float, int, or ndarray
            - If shape is (1,): returns scalar
            - If shape is (1, n): returns 1D array of length n
            - Otherwise: returns the full data array
        
        Examples
        --------
        >>> # Extract scalar from fully reduced series
        >>> scalar = ts.ensemble_rms.time_mean.value  # → float
        >>> 
        >>> # Extract array from partially reduced series  
        >>> array = ts.time_mean.value  # → ndarray(ensemble_shape)
        >>> 
        >>> # Works like pandas .values
        >>> values = ts.value  # → full data array
        """
        data = self._get_not_indexed_data()
        
        # Remove trivial dimensions and extract scalar if possible
        squeezed = np.squeeze(data)
        
        if squeezed.ndim == 0:
            # Scalar (0D array) → Python scalar
            return squeezed.item()
        else:
            return squeezed
    
    @property
    def ensemble_rms(self):
        """
        RMS over ensemble dimensions (chainable property).
        
        Returns a TimeSeries containing the RMS computed across all
        ensemble (non-time) dimensions at each time step.
        The result can be chained with time aggregation operations.
        
        Returns
        -------
        TimeSeries
            Time series of ensemble RMS values (numpy-compatible via __array__)
        
        Notes
        -----
        This property:
        - Works on ALL data (no filtering)
        - Returns TimeSeries (chainable with .time_mean, .time_std, etc.)
        - Is numpy-compatible via __array__ protocol
        
        For filtered data, use upstream filtering:
        ``ts.filter(modes=[2,3,4]).with_times([1,2]).ensemble_rms``
        
        For legacy array return, use: :meth:`get_ensemble_rms` (deprecated)
        
        Examples
        --------
        >>> # Chainable operations
        >>> ts = WavefrontSeries(...)  # shape (n_frames, ny, nx)
        >>> rms_series = ts.ensemble_rms  # shape (n_frames,) - chainable!
        >>> mean_rms = rms_series.time_mean  # scalar - fluent API!
        >>> 
        >>> # Numpy/matplotlib compatible:
        >>> plt.plot(time, rms_series)  # Works via __array__()
        >>> np.mean(rms_series)  # Works!
        >>> 
        >>> # With filtering (upstream):
        >>> mean_rms = ts.filter(modes=[2,3,4]).ensemble_rms.time_mean
        """
        data = self._get_not_indexed_data()
        axis = self._data_sample_axes(data)
        
        if isinstance(data, np.ma.MaskedArray):
            rms_data = np.sqrt(np.ma.mean(np.abs(data)**2, axis=axis))
        else:
            rms_data = np.sqrt(np.mean(np.abs(data)**2, axis=axis))
        
        return self._create_reduced_series(rms_data, operation_name='ensemble_rms')
    
    @property
    def time_mean(self):
        """
        Mean over time dimension (chainable property).
        
        Computes the average across time (axis 0), returning a
        TimeSeries with time_size=1 containing the temporal mean.
        Further operations (like ensemble_rms, ensemble_ptp) can be chained.
        
        Returns
        -------
        TimeSeries
            TimeSeries with shape (1, ...ensemble_shape...) (chainable)
            Use .value to extract the final array/scalar.
        
        Notes
        -----
        Chainable operation: returns TimeSeries (not array).
        For convenient array extraction, use .value:
        ``ts.time_mean.value`` → ndarray or scalar
        
        Examples
        --------
        >>> ts = WavefrontSeries(...)  # shape (n_frames, ny, nx)
        >>> mean_ts = ts.time_mean  # TimeSeries(1, ny, nx) - chainable!
        >>> mean_wf = ts.time_mean.value  # ndarray(ny, nx) - extracted
        >>> 
        >>> # Chaining works both ways now:
        >>> rms_series = ts.ensemble_rms  # (n_frames,)
        >>> mean_rms = rms_series.time_mean.value  # scalar
        >>> 
        >>> # NEW: time-then-ensemble operations
        >>> ptp_map = ts.time_mean.ensemble_ptp.value  # peak-to-valley of long-exposure
        """
        data = self._get_not_indexed_data()
        if isinstance(data, np.ma.MaskedArray):
            mean_data = np.ma.mean(data, axis=0)
        else:
            mean_data = np.mean(data, axis=0)
        
        return self._create_temporal_reduced_series(mean_data, operation_name='time_mean')
    
    @property
    def time_std(self):
        """
        Standard deviation over time dimension (chainable property).
        
        Computes the std across time (axis 0), returning a
        TimeSeries with time_size=1 containing the temporal std.
        Further operations (like ensemble_rms, ensemble_ptp) can be chained.
        
        Returns
        -------
        TimeSeries
            TimeSeries with shape (1, ...ensemble_shape...) (chainable)
            Use .value to extract the final array/scalar.
        
        Notes
        -----
        Chainable operation: returns TimeSeries (not array).
        For convenient array extraction, use .value:
        ``ts.time_std.value`` → ndarray or scalar
        
        Examples
        --------
        >>> ts = WavefrontSeries(...)  # shape (n_frames, ny, nx)
        >>> std_ts = ts.time_std  # TimeSeries(1, ny, nx) - chainable!
        >>> std_wf = ts.time_std.value  # ndarray(ny, nx) - extracted
        >>> 
        >>> # Chaining:
        >>> rms_series = ts.ensemble_rms  # (n_frames,)
        >>> std_rms = rms_series.time_std.value  # scalar
        >>> 
        >>> # NEW: time-then-ensemble operations
        >>> ptp_std = ts.time_std.ensemble_ptp.value  # peak-to-peak temporal variability
        """
        data = self._get_not_indexed_data()
        if isinstance(data, np.ma.MaskedArray):
            std_data = np.ma.std(data, axis=0)
        else:
            std_data = np.std(data, axis=0)
        
        return self._create_temporal_reduced_series(std_data, operation_name='time_std')
    
    @property
    def ensemble_mean(self):
        """
        Mean over ensemble dimensions (chainable property).
        
        Returns
        -------
        TimeSeries
            Mean across ensemble at each time step (chainable)
        
        Examples
        --------
        >>> mean_series = ts.ensemble_mean.value  # Shape: (n_times,)
        >>> overall_mean = ts.ensemble_mean.time_mean.value  # Scalar
        """
        data = self._get_not_indexed_data()
        axis = self._data_sample_axes(data)
        if isinstance(data, np.ma.MaskedArray):
            mean_data = np.ma.mean(data, axis=axis)
        else:
            mean_data = np.mean(data, axis=axis)
        return self._create_reduced_series(mean_data, operation_name='ensemble_mean')
    
    @property
    def ensemble_std(self):
        """
        Standard deviation over ensemble dimensions (chainable property).
        
        Returns
        -------
        TimeSeries
            Std across ensemble at each time step (chainable)
        
        Examples
        --------
        >>> std_series = ts.ensemble_std.value
        >>> mean_std = ts.ensemble_std.time_mean.value
        """
        data = self._get_not_indexed_data()
        axis = self._data_sample_axes(data)
        if isinstance(data, np.ma.MaskedArray):
            std_data = np.ma.std(data, axis=axis)
        else:
            std_data = np.std(data, axis=axis)
        return self._create_reduced_series(std_data, operation_name='ensemble_std')
    
    @property
    def ensemble_median(self):
        """
        Median over ensemble dimensions (chainable property).
        
        Returns
        -------
        TimeSeries
            Median across ensemble at each time step (chainable)
        
        Examples
        --------
        >>> median_series = ts.ensemble_median.value
        >>> time_avg_median = ts.ensemble_median.time_mean.value
        """
        data = self._get_not_indexed_data()
        axis = self._data_sample_axes(data)
        if isinstance(data, np.ma.MaskedArray):
            median_data = np.ma.median(data, axis=axis)
        else:
            median_data = np.median(data, axis=axis)
        return self._create_reduced_series(median_data, operation_name='ensemble_median')
    
    @property
    def ensemble_ptp(self):
        """
        Peak-to-peak over ensemble dimensions (chainable property).
        
        Returns
        -------
        TimeSeries
            Peak-to-peak across ensemble at each time step (chainable)
        
        Examples
        --------
        >>> ptp_series = ts.ensemble_ptp.value
        >>> mean_ptp = ts.ensemble_ptp.time_mean.value
        """
        data = self._get_not_indexed_data()
        axis = self._data_sample_axes(data)
        if isinstance(data, np.ma.MaskedArray):
            ptp_data = np.ma.ptp(data, axis=axis)
        else:
            ptp_data = np.ptp(data, axis=axis)
        return self._create_reduced_series(ptp_data, operation_name='ensemble_ptp')
    
    @property
    def time_median(self):
        """
        Median over time dimension (chainable property).
        
        Returns
        -------
        TimeSeries
            TimeSeries with time_size=1 containing temporal median (chainable)
        
        Examples
        --------
        >>> median_wf = ts.time_median.value
        >>> rms_of_median = ts.time_median.ensemble_rms.value
        """
        data = self._get_not_indexed_data()
        if isinstance(data, np.ma.MaskedArray):
            median_data = np.ma.median(data, axis=0)
        else:
            median_data = np.median(data, axis=0)
        return self._create_temporal_reduced_series(median_data, operation_name='time_median')
    
    @property
    def time_rms(self):
        """
        RMS over time dimension (chainable property).
        
        Returns
        -------
        TimeSeries
            TimeSeries with time_size=1 containing temporal RMS (chainable)
        
        Examples
        --------
        >>> rms_wf = ts.time_rms.value
        >>> max_rms = ts.time_rms.ensemble_ptp.value
        """
        data = self._get_not_indexed_data()
        if isinstance(data, np.ma.MaskedArray):
            rms_data = np.sqrt(np.ma.mean(np.abs(data)**2, axis=0))
        else:
            rms_data = np.sqrt(np.mean(np.abs(data)**2, axis=0))
        return self._create_temporal_reduced_series(rms_data, operation_name='time_rms')
    
    @property
    def time_ptp(self):
        """
        Peak-to-peak over time dimension (chainable property).
        
        Returns
        -------
        TimeSeries
            TimeSeries with time_size=1 containing temporal peak-to-peak (chainable)
        
        Examples
        --------
        >>> ptp_wf = ts.time_ptp.value
        >>> mean_ptp = ts.time_ptp.ensemble_mean.value
        """
        data = self._get_not_indexed_data()
        if isinstance(data, np.ma.MaskedArray):
            ptp_data = np.ma.ptp(data, axis=0)
        else:
            ptp_data = np.ptp(data, axis=0)
        return self._create_temporal_reduced_series(ptp_data, operation_name='time_ptp')

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
            Units are data_unit^2 / Hz (standard PSD units) if applicable.
        
        Notes
        -----
        The PSD is computed using scipy.signal.welch in standard units [data_unit^2/Hz].
        Results are cached unless segment_factor or window changes.
        
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
        # Return PSD in standard units [data_unit^2/Hz] as computed by scipy.signal.welch
        return x.T


# ___oOo___
