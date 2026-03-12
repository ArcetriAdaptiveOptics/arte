
import numpy as np
import astropy.units as u

from arte.utils.help import modify_help
from arte.time_series.time_series import TimeSeries

class MultiTimeSeries(TimeSeries):
    '''
    Join multiple TimeSeries objects with incompatible time sampling.
    
    After being initialized with a list of
    :class:`~arte.time_series.time_series.TimeSeries` objects, which can
    have different ensemble sizes and length, it behaves as if all the data
    was part of a single ensemble. 
    
    **Time operations** (like :attr:`time_mean`, :attr:`time_std`) work in any case.
    
    **Ensemble operations** (like :attr:`ensemble_mean`, :attr:`ensemble_std`) 
    only work if all series have the same sampling rate (``delta_time``), 
    and raise an Exception otherwise. Use :meth:`is_homogeneous` to check 
    compatibility before calling ensemble operations.
    
    Examples
    --------
    >>> # Create multi-series with compatible sampling
    >>> multi = MultiTimeSeries(series1_1khz, series2_1khz)
    >>> multi.is_homogeneous()  # True - same sampling rates
    True
    >>> mean = multi.ensemble_mean.value  # Works!
    >>> 
    >>> # Incompatible sampling raises exception
    >>> multi2 = MultiTimeSeries(series_1khz, series_500hz)
    >>> multi2.ensemble_mean  # Raises Exception - incompatible rates
    >>> 
    >>> # Time operations always work
    >>> time_avg = multi2.time_mean.value  # Always works
    
    Notes
    -----
    Chainable API: ensemble operations return TimeSeries objects that can
    be chained with time operations and vice versa.
    '''
    def __init__(self, *args):
        super().__init__()

        self._series = []
        for v in args:
            self.add_series(v)

    def add_series(self, series):
        '''
        Adds a new series to this MultiTimeSeries instance
        
        Parameters
        ----------
        series: :class:`~arte.time_series.time_series.TimeSeries` or :class:`~arte.time_series.time_series.TimeSeriesWithInterpolation` instance
           the series to be added        
        '''
        self._series.append(series)

    def _get_not_indexed_data(self):
        return np.hstack([v._get_not_indexed_data()
                          for v in self._series])
        
    def is_homogeneous(self, *args, **kwargs):
        '''
        Check if selected series have compatible sampling rates.
        
        Parameters
        ----------
        *args, **kwargs
            Passed to :meth:`get_index_of` for series selection.
        
        Returns
        -------
        bool
            True if all selected series have identical `delta_time`,
            False otherwise.
        
        Notes
        -----
        This check is necessary before calling ensemble-wise operations
        like :attr:`ensemble_mean` or :attr:`ensemble_std`, which
        require uniform time sampling across all series.
        
        Examples
        --------
        >>> multi = MultiTimeSeries(series_1khz, series_500hz)
        >>> multi.is_homogeneous()  # False - different sampling rates
        False
        >>> 
        >>> multi2 = MultiTimeSeries(series1_1khz, series2_1khz)
        >>> multi2.is_homogeneous()  # True - same sampling
        True
        '''
        dt = [x.value for x in self.delta_times(*args, **kwargs)]
        return len(set(dt)) == 1

    def _impersonateDeltaTime(self, *args, **kwargs):
        '''Assume the first delta time of the ones selected by the args'''
        dt = self.delta_times(*args, **kwargs)
        self.delta_time = dt[0]

    def ensemble_size(self):
        '''Returns the total ensemble size'''
        return sum([x.ensemble_size() for x in self._series])

    def delta_times(self, *args, **kwargs):
        '''Returns a vector of delta times'''

        # Known astropy bug (numpy < 1.17): units are lost when using hstack 
        # We remove them before stacking, and add them later
        dt = np.hstack( \
              [np.repeat(x.delta_time.to('s').value, x.ensemble_size())
               for x in self._series])

        dt = dt * u.s

        index = self.get_index_of(*args, **kwargs)
        if index is not None and len(index)>0:
            dt = dt[index]

        return dt

    # Legacy methods (deprecated - use properties instead)
    
    @modify_help(arg_str='[time_idx]')
    def get_ensemble_average(self, *args, times=None, **kwargs):
        '''
        Average across series at each sampling time (legacy method).
        
        .. deprecated::
           Use the `ensemble_mean` property instead. This method will be removed in a future version.
        
        Returns
        -------
        ndarray
            Mean values (not chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates (not homogeneous)
        '''
        import warnings
        warnings.warn(
            "get_ensemble_average() is deprecated. Use the 'ensemble_mean' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().get_ensemble_average(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

    @modify_help(arg_str='[time_idx]')
    def get_ensemble_std(self, *args, times=None, **kwargs):
        '''
        Standard deviation across series at each sampling time (legacy method).
        
        .. deprecated::
           Use the `ensemble_std` property instead. This method will be removed in a future version.
        
        Returns
        -------
        ndarray
            Std values (not chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates (not homogeneous)
        '''
        import warnings
        warnings.warn(
            "get_ensemble_std() is deprecated. Use the 'ensemble_std' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().get_ensemble_std(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

    @modify_help(arg_str='[time_idx]')
    def get_ensemble_median(self, *args, times=None, **kwargs):
        '''
        Median across series at each sampling time (legacy method).
        
        .. deprecated::
           Use the `ensemble_median` property instead. This method will be removed in a future version.
        
        Returns
        -------
        ndarray
            Median values (not chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates (not homogeneous)
        '''
        import warnings
        warnings.warn(
            "get_ensemble_median() is deprecated. Use the 'ensemble_median' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().get_ensemble_median(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

    @modify_help(arg_str='[time_idx]')
    def get_ensemble_rms(self, *args, times=None, **kwargs):
        '''
        Root-Mean-Square across series at each sampling time (legacy method).
        
        .. deprecated::
           Use the `ensemble_rms` property instead. This method will be removed in a future version.
        
        Returns
        -------
        ndarray
            RMS values (not chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates (not homogeneous)
        '''
        import warnings
        warnings.warn(
            "get_ensemble_rms() is deprecated. Use the 'ensemble_rms' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().get_ensemble_rms(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')

    @modify_help(arg_str='[time_idx]')
    def get_ensemble_ptp(self, *args, times=None, **kwargs):
        '''
        Peak-to-peak (max - min) across series at each sampling time (legacy method).
        
        .. deprecated::
           Use the `ensemble_ptp` property instead. This method will be removed in a future version.
        
        Returns
        -------
        ndarray
            Peak-to-peak values (not chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates (not homogeneous)
        '''
        import warnings
        warnings.warn(
            "get_ensemble_ptp() is deprecated. Use the 'ensemble_ptp' property instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.is_homogeneous(*args, **kwargs):
            self._impersonateDeltaTime(*args, **kwargs) 
            return super().get_ensemble_ptp(*args, times=times, **kwargs)
        else:
            raise Exception('Data series cannot be combined')
    
    # Chainable properties (override base class to add homogeneity check)
    
    @property
    def ensemble_mean(self):
        """
        Mean over ensemble dimensions (chainable property).
        
        Computes the mean across all series at each time step.
        Requires all series to have compatible sampling rates.
        
        Returns
        -------
        TimeSeries
            Mean across ensemble at each time step (chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates.
            Use :meth:`is_homogeneous` to check compatibility first.
        
        Examples
        --------
        >>> # Check compatibility before using
        >>> if multi.is_homogeneous():
        ...     mean = multi.ensemble_mean.value
        >>> 
        >>> # Chain with time operations
        >>> overall_mean = multi.ensemble_mean.time_mean.value
        
        See Also
        --------
        is_homogeneous : Check if series have compatible sampling rates
        ensemble_std : Standard deviation across ensemble
        ensemble_median : Median across ensemble
        """
        if not self.is_homogeneous():
            raise Exception('Data series cannot be combined - incompatible sampling rates. '
                          'Use is_homogeneous() to check compatibility.')
        self._impersonateDeltaTime()
        return super().ensemble_mean
    
    @property
    def ensemble_std(self):
        """
        Standard deviation over ensemble dimensions (chainable property).
        
        Computes the std across all series at each time step.
        Requires all series to have compatible sampling rates.
        
        Returns
        -------
        TimeSeries
            Std across ensemble at each time step (chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates.
            Use :meth:`is_homogeneous` to check compatibility first.
        
        Examples
        --------
        >>> if multi.is_homogeneous():
        ...     std = multi.ensemble_std.value
        >>> 
        >>> # Chain with time operations
        >>> mean_std = multi.ensemble_std.time_mean.value
        
        See Also
        --------
        is_homogeneous : Check if series have compatible sampling rates
        ensemble_mean : Mean across ensemble
        """
        if not self.is_homogeneous():
            raise Exception('Data series cannot be combined - incompatible sampling rates. '
                          'Use is_homogeneous() to check compatibility.')
        self._impersonateDeltaTime()
        return super().ensemble_std
    
    @property
    def ensemble_median(self):
        """
        Median over ensemble dimensions (chainable property).
        
        Computes the median across all series at each time step.
        Requires all series to have compatible sampling rates.
        
        Returns
        -------
        TimeSeries
            Median across ensemble at each time step (chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates.
            Use :meth:`is_homogeneous` to check compatibility first.
        
        Examples
        --------
        >>> if multi.is_homogeneous():
        ...     median = multi.ensemble_median.value
        >>> 
        >>> # Chain with time operations
        >>> median_avg = multi.ensemble_median.time_mean.value
        
        See Also
        --------
        is_homogeneous : Check if series have compatible sampling rates
        ensemble_mean : Mean across ensemble
        """
        if not self.is_homogeneous():
            raise Exception('Data series cannot be combined - incompatible sampling rates. '
                          'Use is_homogeneous() to check compatibility.')
        self._impersonateDeltaTime()
        return super().ensemble_median
    
    @property
    def ensemble_rms(self):
        """
        RMS over ensemble dimensions (chainable property).
        
        Computes the RMS across all series at each time step.
        Requires all series to have compatible sampling rates.
        
        Returns
        -------
        TimeSeries
            RMS across ensemble at each time step (chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates.
            Use :meth:`is_homogeneous` to check compatibility first.
        
        Examples
        --------
        >>> if multi.is_homogeneous():
        ...     rms = multi.ensemble_rms.value
        >>> 
        >>> # Chain with time operations
        >>> mean_rms = multi.ensemble_rms.time_mean.value
        
        See Also
        --------
        is_homogeneous : Check if series have compatible sampling rates
        ensemble_mean : Mean across ensemble
        """
        if not self.is_homogeneous():
            raise Exception('Data series cannot be combined - incompatible sampling rates. '
                          'Use is_homogeneous() to check compatibility.')
        self._impersonateDeltaTime()
        return super().ensemble_rms
    
    @property
    def ensemble_ptp(self):
        """
        Peak-to-peak over ensemble dimensions (chainable property).
        
        Computes the peak-to-peak range across all series at each time step.
        Requires all series to have compatible sampling rates.
        
        Returns
        -------
        TimeSeries
            Peak-to-peak across ensemble at each time step (chainable)
        
        Raises
        ------
        Exception
            If series have incompatible sampling rates.
            Use :meth:`is_homogeneous` to check compatibility first.
        
        Examples
        --------
        >>> if multi.is_homogeneous():
        ...     ptp = multi.ensemble_ptp.value
        >>> 
        >>> # Chain with time operations
        >>> mean_ptp = multi.ensemble_ptp.time_mean.value
        
        See Also
        --------
        is_homogeneous : Check if series have compatible sampling rates
        """
        if not self.is_homogeneous():
            raise Exception('Data series cannot be combined - incompatible sampling rates. '
                          'Use is_homogeneous() to check compatibility.')
        self._impersonateDeltaTime()
        return super().ensemble_ptp

# ___oOo___