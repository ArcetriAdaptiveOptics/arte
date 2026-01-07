import collections


class _AnalyzerStub(list):

    def __getattr__(self, name):
        # Necessary to allow casting by np.array()
        if name.startswith('__array'):
            raise AttributeError

        if hasattr(self[0], name):
            return _AnalyzerStub([getattr(x, name) for x in self])
        else:
            raise AttributeError

    def __call__(self, *args, **kwargs):
        return _AnalyzerStub([x(*args, **kwargs) for x in self])


class BaseAnalyzerSet():
    """Collection of Analyzer objects for batch analysis.
    
    AnalyzerSet manages multiple Analyzer instances across a range of tags,
    enabling batch analysis and comparison of datasets. It provides:
    
    - Lazy instantiation of analyzers (created only when accessed)
    - Iteration over all analyzers in the set
    - Array-like access by tag or index
    - Automatic attribute forwarding to all analyzers
    
    Parameters
    ----------
    from_or_list : str or list
        Either a single tag, a list of tags, or a start tag
    to : str, optional
        End tag when using a date range. If None, from_or_list must be
        a list of tags
    recalc : bool, optional
        If True, force recalculation of cached data for all analyzers
        (default: False)
    file_walker : AbstractFileNameWalker instance
        File walker to find tags and their data (required keyword argument)
    analyzer_type : BaseAnalyzer class
        Analyzer class to instantiate (not an instance). Must implement
        a get() classmethod (required keyword argument)
    
    Attributes
    ----------
    tag_list : list of str
        Sorted list of tags in this set
    
    Examples
    --------
    >>> # Create analyzer set from tag range
    >>> aset = MyAnalyzerSet(
    ...     '20240101_000000', '20240101_235959',
    ...     file_walker=my_walker,
    ...     analyzer_type=MyAnalyzer
    ... )
    
    >>> # Remove invalid tags
    >>> aset.remove_invalids()
    
    >>> # Iterate over analyzers
    >>> for analyzer in aset:
    ...     print(analyzer.residual_modes.time_std())
    
    >>> # Access by tag
    >>> analyzer = aset['20240101_120000']
    
    >>> # Attribute forwarding (returns list of results)
    >>> all_std = aset.residual_modes.time_std()
    
    Notes
    -----
    Attribute access on AnalyzerSet returns an _AnalyzerStub that forwards
    the same attribute/method call to all analyzers and returns the results
    as a list.
    """
    def __init__(self, from_or_list, to=None, recalc=False, *,
                       file_walker, analyzer_type):
        '''
        This constructor only builds the list of tags but does not allocate any Analyzer.

        Parameters
        ----------
        from_or_list: str or list
            either a single tag, or a list of tags
        to: str, optional
            sigle tag, or None
        recalc: bool
            if True, all analyzers will be recalculated (lazy recalc, only when actually accessed)
        file_walker: BaseFileWalker or derived class instance
            file walker used to find tag data (required keyword argument)
        analyzer_type: BaseAnalyzer or derived class (not an instance).
            analyzer type to instance for each tag. Must define a get() classmethod (required keyword argument).
        '''
        self._analyzer_args = []
        self._analyzer_kwargs = {}
        self._file_walker = file_walker
        self._analyzer_type = analyzer_type

        if isinstance(from_or_list, collections.abc.Sequence) and not isinstance(from_or_list, str):
            self.tag_list = sorted(from_or_list)
        else:
            self.tag_list = sorted(self._file_walker.find_tag_between_dates(str(from_or_list), str(to)))
        self._init_recalcs = {k: recalc for k in self.tag_list}

    def remove_invalids(self):
        '''
        Remove tags that evaluate to NotAvailable when created
        '''
        newtags = []
        for tag in self.tag_list:
            ee = self.get(tag, recalc=self._init_recalcs[tag])
            if str(ee) != 'NA':
                newtags.append(tag)
        self.tag_list = newtags

    def set_analyzer_args(self, *args, **kwargs):
        '''Set the additional arguments to pass to Analyzers' contructors'''
        self._analyzer_args = args
        self._analyzer_kwargs = kwargs

    def __iter__(self):
        for tag in self.tag_list:
            yield self.get(tag)

    def get(self, tag, recalc=False):
        '''Returns the Analyzer instance for this tag'''
        my_recalc = self._init_recalcs[tag] or recalc
        self._init_recalcs[tag] = False
        return self._analyzer_type.get(tag, *self._analyzer_args,
                                    recalc=my_recalc, **self._analyzer_kwargs)

    def __getitem__(self, idx_or_tag):
        if isinstance(idx_or_tag, int):
            return self.get(self.tag_list[idx_or_tag])
        else:
            return self.get(idx_or_tag)

    def append(self, tag):
        self.tag_list.append(tag)

    def insert(self, idx, tag):
        self.tag_list.insert(idx, tag)

    def remove(self, tag):
        _= self.tag_list.remove(tag)

    def __len__(self):
        return len(self.tag_list)

    def _apply(self, func_name, *args, **kwargs):

        for tag in self.tag_list:
            getattr(self.get(tag), func_name).__call__(*args, **kwargs)

    def _apply_w_args(self, func_name, args_list, kwargs_list):

        for tag,args,kwargs in zip(self.tag_list, args_list, kwargs_list):
            getattr(self.get(tag), func_name).__call__(*args, **kwargs)

    def generate_tags(self):
        for tag in self.tag_list:
            yield self.get(tag)

    def __getattr__(self, attrname):
        if hasattr(self.get(self.tag_list[0]), attrname):
            return _AnalyzerStub([getattr(self.get(tag), attrname) for tag in self.tag_list])
        else:
            raise AttributeError

    def wiki(self):
        '''Print wiki info on stdout'''
        for i, ee in enumerate(self.generate_tags()):
            ee.wiki(header = (i == 0))

    
