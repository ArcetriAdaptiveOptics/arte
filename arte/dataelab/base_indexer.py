from itertools import chain  # List flattening

class BaseIndexer():
    """Base indexer for flexible data subset selection.
    
    Indexers provide a flexible interface for selecting subsets of data
    from time series. They process various argument formats:
    
    - Positional arguments for direct element selection
    - Keyword arguments with customizable names
    - Slice syntax for range selection
    
    Parameters
    ----------
    single_kw : str or list of str, optional
        Additional keyword names for single element selection.
        Default keywords: 'element', 'elements'
    from_kw : str or list of str, optional
        Additional keyword names for start of range.
        Default keywords: 'from_element', 'first'
    to_kw : str or list of str, optional
        Additional keyword names for end of range.
        Default keywords: 'to_element', 'last'
    
    Examples
    --------
    >>> indexer = BaseIndexer(single_kw='mode', from_kw='start')
    
    >>> # Select single elements
    >>> idx = indexer.process_args(5)              # Returns 5
    >>> idx = indexer.process_args(element=5)      # Returns 5
    >>> idx = indexer.process_args(mode=5)         # Returns 5
    
    >>> # Select element range
    >>> idx = indexer.process_args(first=10, last=20)  # Returns slice(10, 20)
    >>> idx = indexer.process_args(start=10)           # Returns slice(10, None)
    
    >>> # Select list of elements
    >>> idx = indexer.process_args(elements=[1,5,9])   # Returns [1, 5, 9]
    
    Notes
    -----
    This class is typically used as a base for specialized indexers
    that understand data structure (e.g., x/y slopes, quadrants).
    
    See Also
    --------
    arte.time_series.indexer.Indexer : Advanced indexer with more features
    """

    def __init__(self, single_kw=None, from_kw=None, to_kw=None):
        self.single_kw = ['element', 'elements']
        self.from_kw= ['from_element', 'first']
        self.to_kw = ['to_element', 'last']

        if single_kw:
            self.single_kw += [single_kw]
            self.single_kw = list(chain.from_iterable(self.single_kw))
        if from_kw:
            self.from_kw += [from_kw]
            self.from_kw = list(chain.from_iterable(self.from_kw))
        if to_kw:
            self.to_kw += [to_kw]
            self.to_kw = list(chain.from_iterable(self.to_kw))

    def process_args(self, *args, **kwargs):
        '''
        default: all elements
        element = single element
        elements = list of elements
        from_element = first element
        to_element = last element
        '''
        elements = None
        from_element = None
        to_element = None
        if len(args) > 0:
            elements = args
        else:
            for k, v in kwargs.items():
                if k in self.single_kw:
                    elements = v
                if k in self.from_kw:
                    from_element = v
                if k in self.to_kw:
                    to_element = v
        if elements is not None:
            return elements
        return slice(from_element, to_element)