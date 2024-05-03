import abc
from arte.dataelab.tag import Tag


class _AnalyzerStub(list):

    def __getattr__(self, name):
        if hasattr(self[0], name):
            return _AnalyzerStub([getattr(x, name) for x in self])
        else:
            raise AttributeError

    def __call__(self, *args, **kwargs):
        return _AnalyzerStub([x(*args, **kwargs) for x in self])


class BaseAnalyzerSet():
    '''
    Analyzer set. Holds a list of Analyzer objects
    '''
    def __init__(self, from_or_list, to=None, recalc=False, skip_invalid=True):
        '''
        from_or_list: either a single tag, or a list of tags
        to: sigle tag, or None
        analyzer_pool: AnalyzerPool instance from which Analyzer instances can be get()ted
        recalc: if True, all analyzers will be recalculated (lazy recalc, only when actually accessed)
        skip_invalid: if True (default), skip analyzers that throw exceptions during initialization
        '''
        if isinstance(from_or_list, str):
            if to is None:
                to = Tag.create_tag()
            self.tag_list= self._get_file_walker().find_tag_between_dates(from_or_list, str(to))
        else:
            self.tag_list= sorted(from_or_list)

        if skip_invalid:
            newtags = []
            for tag in self.tag_list:
                ee = self.get(tag, recalc=recalc)
                if str(ee) != 'NA':
                    newtags.append(tag)
            self.tag_list = newtags

        if recalc and not skip_invalid:
            for tag in self.tag_list:
                _ = self.get(tag, recalc=recalc)

    @abc.abstractmethod
    def _get_file_walker(self):
        '''Return a FileWalker instance'''
        pass

    @abc.abstractmethod
    def _get_type(self):
        '''Return the Analyzer class type'''
        pass

    def __iter__(self):
        for tag in self.tag_list:
            yield self.get(tag)

    def get(self, tag, recalc=False):
        '''Returns the Analyzer instance for this tag'''
        return self._get_type()._get(tag, recalc=recalc)

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
        conf=None
        for i, ee in enumerate(self.generate_tags()):
            ee.wiki(header = (i == 0))

    