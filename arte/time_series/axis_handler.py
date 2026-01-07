'''
Axis transposition handler for multi-dimensional time series data.

This module provides utilities for managing named axes in time series arrays
and performing axis reordering based on axis names rather than numeric indices.
'''

class AxisHandler():
    '''
    Handle axis transpositions in multi-dimensional arrays using named axes.
    
    This class enables intuitive axis reordering for time series data with
    multiple ensemble dimensions. Instead of using numeric indices, axes can
    be referenced by meaningful names (e.g., 'subaperture', 'wavelength', 'mode').
    
    Parameters
    ----------
    axes : sequence of str or str, optional
        Names for each axis in the data (excluding time axis, which is always first).
        If None, no axis transposition will be possible.
        If a single string, wraps it in a tuple.
    
    Examples
    --------
    >>> # Define 3D time series: (time, subaperture, wavelength)
    >>> handler = AxisHandler(axes=['subaperture', 'wavelength'])
    >>> 
    >>> # Transpose to (subaperture, wavelength, time)
    >>> data_transposed = handler.transpose(data, ['subaperture', 'wavelength'])
    >>> 
    >>> # Reorder axes: (time, wavelength, subaperture)
    >>> data_reordered = handler.transpose(data, ['wavelength', 'subaperture'])
    '''

    def __init__(self, axes=None):
        '''
        Initialize axis handler.

        Parameters
        ----------
        axes : sequence of str or str, optional
            Names for data axes. If None, no transposition possible.
        '''
        if axes is None:
            self._axes = ()
        elif isinstance(axes, str):
            self._axes = (axes,)
        else:
            self._axes = tuple(axes)

    def transpose(self, data, axes=None):
        '''
        Transpose array axes according to named axis order.
        
        Parameters
        ----------
        data : ndarray
            Array to transpose. First dimension (time) is not affected.
        axes : sequence of str or str, optional
            Desired axis order using axis names defined during initialization.
            If None, no transposition is performed.
        
        Returns
        -------
        ndarray
            Transposed array with axes reordered as specified.
        
        Raises
        ------
        ValueError
            If any requested axis name was not defined during initialization.
        
        Examples
        --------
        >>> handler = AxisHandler(axes=['x', 'y', 'wavelength'])
        >>> data = np.random.rand(100, 10, 20, 5)  # (time, x, y, wavelength)
        >>> 
        >>> # Reorder to (time, wavelength, x, y)
        >>> data_new = handler.transpose(data, ['wavelength', 'x', 'y'])
        >>> data_new.shape  # (100, 5, 10, 20)
        '''
        if axes is not None:
            if type(axes) is str:
                axes = [axes]
            # Use an explicit for loop in order to capture an erroneus axis name
            idx = []
            try:
                for ax in axes:
                    idx.append(self._axes.index(ax))
            except ValueError:
                raise ValueError(f'Axis {ax} not found')
            data = data.transpose(idx)
        return data

    def axes(self):
        return self._axes
