import numpy as np


class Indexer(object):
    '''
    Base indexer class for accessing subsets of time series ensemble data.
    
    Provides utility methods for indexing 2D coordinate data (x, y) which is
    common in wavefront sensor measurements and other spatial data.
    
    This class serves as a foundation for specialized indexers like
    :class:`ModeIndexer`, :class:`DefaultIndexer`, and :class:`RowColIndexer`.
    '''

    def __init__(self):
        self.dicCoord = {'x': 0, 'y': 1}

    def _xy_index(self, coord):
        if not hasattr(coord, '__iter__'):
            coord = [coord]
        indexCoord = [self.dicCoord[x] for x in coord]
        return indexCoord

    def interleaved_xy(self, *args, **kwargs):
        '''
        Create slice for interleaved x/y data layout.
        
        This indexer is designed for wavefront sensor slopes stored in
        interleaved format: [x0, y0, x1, y1, x2, y2, ...]
        
        Parameters
        ----------
        *args : str or int, optional
            If 'x': select x coordinates (indices 0, 2, 4, ...)
            If 'y': select y coordinates (indices 1, 3, 5, ...)
            If int: select single element at that index
            If not provided: select all elements
        coord, coords, axis : {'x', 'y'}, optional
            Alternative keyword-based coordinate selection.
        
        Returns
        -------
        slice
            Python slice object for indexing the requested coordinates.
        
        Examples
        --------
        >>> indexer = Indexer()
        >>> slopes = np.array([sx0, sy0, sx1, sy1, sx2, sy2])  # interleaved
        >>> 
        >>> # Select all x slopes
        >>> x_slopes = slopes[indexer.interleaved_xy('x')]  # [sx0, sx1, sx2]
        >>> 
        >>> # Select all y slopes  
        >>> y_slopes = slopes[indexer.interleaved_xy(coord='y')]  # [sy0, sy1, sy2]
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if v == 'x':
                return slice(0, None, 2)
            elif v == 'y':
                return slice(1, None, 2)
            else:
                raise Exception("%s not in accepted"
                                " values ('x', 'y')" % (v))

        # Since we are here, no keywords were specified
        if len(args) == 0:
            return slice(0, None, 1)
        elif len(args) == 1:
            if args[0] == 'x':
                return slice(0, None, 2)
            elif args[0] == 'y':
                return slice(1, None, 2)
            elif isinstance(args[0], int):
                return slice(args[0], args[0]+1)
        else:
            raise Exception("%s not in accepted"
                            " values ('x', 'y')" % str(v))

    def sequential_xy(self, maxindex, *args, **kwargs):
        '''
        Create slice for sequential x/y data layout.
        
        This indexer is for data stored with all x coordinates followed by
        all y coordinates: [x0, x1, x2, ..., y0, y1, y2, ...]
        
        Parameters
        ----------
        maxindex : int
            Total number of elements (length of the array to index).
        *args : str or int, optional
            If 'x': select x coordinates (first half)
            If 'y': select y coordinates (second half)
            If int: select single element at that index
            If not provided: select all elements
        coord, coords, axis : {'x', 'y'}, optional
            Alternative keyword-based coordinate selection.
        
        Returns
        -------
        slice
            Python slice object for indexing the requested coordinates.
        
        Examples
        --------
        >>> indexer = Indexer()
        >>> slopes = np.array([sx0, sx1, sx2, sy0, sy1, sy2])  # sequential
        >>> 
        >>> # Select all x slopes
        >>> x_slopes = slopes[indexer.sequential_xy(6, 'x')]  # [sx0, sx1, sx2]
        >>> 
        >>> # Select all y slopes
        >>> y_slopes = slopes[indexer.sequential_xy(6, coord='y')]  # [sy0, sy1, sy2]
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if v == 'x':
                return slice(0, maxindex//2)
            elif v == 'y':
                return slice(maxindex//2, None)
            else:
                raise Exception("%s not in accepted"
                                " values ('x', 'y')" % (v))

        # Since we are here, no keywords were specified
        if len(args) == 0:
            return slice(0, None, 1)
        elif len(args) == 1:
            if args[0] == 'x':
                return slice(0, maxindex//2)
            elif args[0] == 'y':
                return slice(maxindex//2, None)
            elif isinstance(args[0], int):
                return slice(args[0], args[0]+1)
        else:
            raise Exception("%s not in accepted"
                            " values ('x', 'y')" % str(v))
        
    def xy(self, *args, **kwargs):
        '''
        default: coord
        Accepted keywords:
        coord= x and/or y
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        coord = ['x', 'y']
        if len(args) > 0:
            coord = args[1]
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if k == 'coord' or k == 'coords' or k == 'axis':
                coord = v
        return self._xy_index(coord)

    # TODO add boundary checks with maxrange on args
    @staticmethod
    def myrange(*args, maxrange):
        
        if len(args)==0:
            return range(maxrange)
        if len(args)==1:
            return range(args[0])
        if len(args)==2:
            return range(args[0], args[1])
        if len(args)==3:
            return range(args[0], args[1], args[2])
        raise ValueError
            
class ModeIndexer(Indexer):
    '''
    Indexer for modal decomposition data (Zernike, KL modes, etc.).
    
    Provides convenient indexing for time series containing modal coefficients,
    allowing selection by mode number, mode ranges, or lists of specific modes.
    
    Parameters
    ----------
    max_mode : int, optional
        Maximum mode number available in the data. Used as default upper bound
        for range selections.
    
    Examples
    --------
    >>> # For Zernike coefficients time series with 100 modes
    >>> indexer = ModeIndexer(max_mode=100)
    >>> 
    >>> # Select all modes
    >>> all_modes = indexer.modes()  # [0, 1, 2, ..., 99]
    >>> 
    >>> # Select mode range (tip, tilt, focus, astig)
    >>> low_order = indexer.modes(from_mode=2, to_mode=6)  # [2, 3, 4, 5]
    >>> 
    >>> # Select specific modes
    >>> selected = indexer.modes(modes=[2, 5, 10])  # [2, 5, 10]
    '''

    def __init__(self, max_mode=None):
        super().__init__()
        self._max_mode = max_mode        

    def modes(self, *args, **kwargs):
        '''
        Generate mode indices for selection.
        
        Parameters
        ----------
        *args : int or array-like, optional
            Single mode number or list of mode numbers to select.
        mode, modes : int or array-like, optional
            Alternative keyword-based mode selection.
        from_mode : int, optional
            Starting mode number for range selection. Default is 0.
        to_mode : int, optional
            Ending mode number (exclusive) for range selection.
            If not specified, uses `max_mode` from initialization.
        
        Returns
        -------
        int, list, or ndarray
            Mode index/indices for array indexing.
        
        Raises
        ------
        Exception
            If `to_mode` is not specified and `max_mode` was not set
            during initialization.
        
        Examples
        --------
        >>> indexer = ModeIndexer(max_mode=50)
        >>> 
        >>> # All modes
        >>> indexer.modes()  # array([0, 1, 2, ..., 49])
        >>> 
        >>> # Single mode
        >>> indexer.modes(mode=5)  # 5
        >>> 
        >>> # Mode range (e.g., low-order Zernikes)
        >>> indexer.modes(from_mode=2, to_mode=11)  # array([2, 3, ..., 10])
        >>> 
        >>> # Specific modes
        >>> indexer.modes(modes=[2, 3, 4, 10, 15])  # [2, 3, 4, 10, 15]
        '''
        modes = None
        from_mode = None
        to_mode = None
        if (len(args) > 0) & (len(args) < 2):
            modes = args[0]
        for k, v in kwargs.items():
            if k == 'modes' or k == 'mode':
                modes = v
            if k == 'from_mode':
                from_mode = v
            if k == 'to_mode':
                to_mode = v
        if modes is not None:
            return modes
        if from_mode is None:
            from_mode = 0
        if to_mode is None:
            to_mode = self._max_mode
        if to_mode is None:
            raise Exception("to_mode was not specified and max_mode was not set "
                            "when initializing the ModeIndexer, cannot continue") 
        return np.arange(from_mode, to_mode)


class DefaultIndexer(Indexer):
    '''
    Generic element indexer for uniform 1D ensemble data.
    
    Provides flexible indexing for time series where ensemble elements
    are not organized in a special structure (modes, rows/cols, etc.)
    but simply form a flat list.
    
    Examples
    --------
    >>> indexer = DefaultIndexer()
    >>> 
    >>> # Select all elements
    >>> indexer.elements()  # slice(None, None, None)
    >>> 
    >>> # Select single element
    >>> indexer.elements(22)  # 22
    >>> 
    >>> # Select list of elements
    >>> indexer.elements([2, 5, 10])  # [2, 5, 10]
    >>> 
    >>> # Select range
    >>> indexer.elements(from_element=10, to_element=20)  # slice(10, 20, None)
    '''

    def __init__(self):
        super().__init__()

    def elements(self, *args, **kwargs):
        '''
        Generate element indices for selection.
        
        Parameters
        ----------
        *args : int, list, or array-like, optional
            Single element index or list of indices to select.
        element, elements : int, list, or array-like, optional
            Alternative keyword-based element selection.
        from_element : int, optional
            Starting element index for range selection.
        to_element : int, optional
            Ending element index for range selection.
        
        Returns
        -------
        int, list, or slice
            Index/indices for array selection.
        
        Examples
        --------
        >>> indexer = DefaultIndexer()
        >>> 
        >>> # All elements (no filtering)
        >>> indexer.elements()  # slice(None, None)
        >>> 
        >>> # Single element
        >>> indexer.elements(5)  # 5
        >>> 
        >>> # List of elements
        >>> indexer.elements([1, 3, 5, 7])  # [1, 3, 5, 7]
        >>> 
        >>> # Range from index 10 onwards
        >>> indexer.elements(from_element=10)  # slice(10, None)
        '''
        elements = None
        from_element = None
        to_element = None
        if (len(args) > 0) & (len(args) < 2):
            elements = args[0]
        for k, v in kwargs.items():
            if k == 'elements' or k == 'element':
                elements = v
            if k == 'from_element':
                from_element = v
            if k == 'to_element':
                to_element = v
        if elements is not None:
            return elements
        return slice(from_element, to_element)


class RowColIndexer(Indexer):
    '''
    Indexer for 2D array data organized in rows and columns.
    
    Useful for time series data with natural 2D spatial structure,
    such as deformable mirror actuators arranged in a grid or
    detector pixels.
    
    Examples
    --------
    >>> indexer = RowColIndexer()
    >>> 
    >>> # Select specific rows and columns
    >>> idx = indexer.rowcol(rows=5, cols=slice(0, 10))
    >>> data_subset = data[:, idx[0], idx[1]]  # time x rows x cols
    >>> 
    >>> # Using from/to syntax
    >>> idx = indexer.rowcol(row_from=10, row_to=20, col_from=5, col_to=15)
    '''

    def __init__(self):
        super().__init__()

    def rowcol(self, *args, **kwargs):
        '''
        Create row and column indices for 2D data selection.
        
        Parameters
        ----------
        *args : int, slice, or array-like
            Positional arguments for row and column selection:
            - If 1 arg: used for rows, all columns selected
            - If 2 args: (rows, cols)
        rows : int, slice, or array-like, optional
            Row index/indices to select.
        cols : int, slice, or array-like, optional
            Column index/indices to select.
        row_from, row_to, row_step : int, optional
            Convenience parameters for building row slice.
        col_from, col_to, col_step : int, optional
            Convenience parameters for building column slice.
        
        Returns
        -------
        tuple of (row_index, col_index)
            Indices suitable for 2D array indexing.
        
        Raises
        ------
        IndexError
            If unsupported keywords or too many positional arguments provided.
        
        Examples
        --------
        >>> indexer = RowColIndexer()
        >>> 
        >>> # Select row 5, columns 0-10
        >>> rows, cols = indexer.rowcol(rows=5, col_from=0, col_to=10)
        >>> 
        >>> # Using positional arguments
        >>> rows, cols = indexer.rowcol(slice(10, 20), slice(5, 15))
        >>> 
        >>> # Select all rows, specific column range with step
        >>> rows, cols = indexer.rowcol(col_from=0, col_to=100, col_step=2)
        '''
        rows = slice(None, None, None)
        cols = slice(None, None, None)
        if len(args) == 1:
            rows = args[0]
        elif len(args) == 2:
            rows, cols = args
        elif len(args) > 2:
            raise IndexError('Unsupported rowcol index')
        
        accepted = 'rows cols row_from col_from row_to col_to row_step col_step'.split()
        for kw in kwargs:
            if kw not in accepted:
                raise IndexError(f'Unsupported index keyword {kw}')

            if kw == 'rows':
                rows = kwargs['rows']
            if kw == 'cols':
                cols = kwargs['cols']
            if kw == 'row_from':
                rows = slice(kwargs['row_from'], rows.stop, rows.step)
            if kw == 'col_from':
                cols = slice(kwargs['col_from'], cols.stop, cols.step)
            if kw == 'row_to':
                rows = slice(rows.start, kwargs['row_to'], rows.step)
            if kw == 'col_to':
                cols = slice(cols.start, kwargs['col_to'], cols.step)
            if kw == 'row_step':
                rows = slice(rows.start, rows.stop, kwargs['row_step'])
            if kw == 'col_step':
                cols = slice(cols.start, cols.stop, kwargs['col_step'])

        return (rows, cols)
