

import os
import re
import numpy as np

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
    def __init__(self, from_or_list, to, analyzer_pool, recalc=False, skip_invalid=True):
        '''
        from_or_list: either a single tag, or a list of tags
        to: sigle tag, or None
        analyzer_pool: AnalyzerPool instance from which Analyzer instances can be get()ted
        recalc: if True, all analyzers will be recalculated (lazy recalc, only when actually accessed)
        skip_invalid: if True (default), skip analyzers that throw exceptions during initialization
        '''
        self._pool = analyzer_pool

        if isinstance(from_or_list, str):
            if to is None:
                to = Tag.create_tag()
            self.tag_list= self.find_tag_between_dates(from_or_list, to)
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

    def __iter__(self):
        for tag in self.tag_list:
            yield self.get(tag)

    def get(self, tag, recalc=False):
        return self._pool.get(tag, recalc=recalc)

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
        for ee in self.generate_tags():
            ee.wiki(header = (ee.configuration() != conf))
            conf = ee.tconfiguration()

    def find_tag_between_dates(self, tag_start, tag_stop):
        day_start= Tag(tag_start).get_day_as_string()
        day_stop= Tag(tag_stop).get_day_as_string()
        snapshot_root_dir= self._pool.conf().snapshot_root_dir()
        days=[]
        for x in os.listdir(snapshot_root_dir):
            if os.path.isdir(os.path.join(snapshot_root_dir, x)):
                days.append(x)
        days= np.sort(days)
        days= days[days<=day_stop]
        days= days[days>=day_start]
        tags=[]
        r = re.compile('^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]$')
        for day in days:
            li= os.listdir(os.path.join(snapshot_root_dir, day))
            for l in filter(r.match, li):     # filter out everything is not a standard tracking number
                if os.path.isdir(os.path.join(snapshot_root_dir, day, l)):
                    if tag_start <= l <= tag_stop:
                        tags.append(l)
        return sorted(tags)
    