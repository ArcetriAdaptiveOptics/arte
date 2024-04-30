

import os
import numpy as np

from arte.dataelab.tag import Tag


class AnalyzerStub(list):

    def __getattr__(self, name):
        if hasattr(self[0], name):
            return AnalyzerStub([getattr(x, name) for x in self])
        else:
            raise AttributeError

    def __call__(self, *args, **kwargs):
        return AnalyzerStub([x(*args, **kwargs) for x in self])


class BaseAnalyzerSet():

    def __init__(self, from_or_list, to, analyzer_pool, recalc=False, skip_invalid=True):

        self._pool = analyzer_pool

        if isinstance(from_or_list, str):
            if to is None:
                to= Tag.createTag()
            self.tagList= self._findTagBetweenDates(from_or_list, to)
        else:
            self.tagList= sorted(from_or_list)

        if skip_invalid:
            newtags = []
            for tag in self.tagList:
                ee = self.get(tag, recalc=recalc)
                if str(ee) != 'NA':
                    newtags.append(tag)
            self.tagList = newtags

        if recalc:
            for tag in self.tagList:
                _ = self.get(tag, recalc=recalc)

    def __iter__(self):
        for tag in self.tagList:
            yield self.get(tag)

    def get(self, tn, recalc=False):
        return self._pool.get(tn, recalc=recalc)

    def __getitem__(self, idx_or_tn):
        if isinstance(idx_or_tn, int):
            return self.get(self.tagList[idx_or_tn])
        else:
            return self.get(idx_or_tn)

    def append(self, tn):
        self.tagList.append(tn)

    def insert(self, idx, tn):
        self.tagList.insert(idx, tn)

    def remove(self, tn):
        _= self.tagList.remove(tn)

    def __len__(self):
        return len(self.tagList)

    def _apply(self, func_name, *args, **kwargs):

        for tag in self.tagList:
            getattr(self.get(tag), func_name).__call__(*args, **kwargs)

    def _apply_w_args(self, func_name, args_list, kwargs_list):

        for tag,args,kwargs in zip(self.tagList, args_list, kwargs_list):
            getattr(self.get(tag), func_name).__call__(*args, **kwargs)

    def TNs(self):
        for tag in self.tagList:
            yield self.get(tag)

    def __getattr__(self, attrname):
        if hasattr(self.get(self.tagList[0]), attrname):
            return AnalyzerStub([getattr(self.get(tag), attrname) for tag in self.tagList])
        else:
            raise AttributeError

    def wiki(self):
        conf=None
        for ee in self.TNs():
            ee.wiki(header = (ee.tn_conf != conf))
            conf = ee.tn_conf

    def _findTagBetweenDates(self, tagStart, tagStop):
        dayStart= Tag(tagStart).get_day_as_string()
        dayStop= Tag(tagStop).get_day_as_string()
        snapshot_root_dir= self._pool.conf().snapshot_root_dir()
        days=[]
        for x in os.listdir(snapshot_root_dir):
            if os.path.isdir(os.path.join(snapshot_root_dir, x)):
                days.append(x)
        days= np.sort(days)
        days= days[days<=dayStop]
        days= days[days>=dayStart]
        tns=[]
        import re
        r = re.compile('^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]$')
        for day in days:
            li= os.listdir(os.path.join(snapshot_root_dir, day))
            for l in filter(r.match,li):     # filter out everything is not a standard tracking number
                if os.path.isdir(os.path.join(snapshot_root_dir, day, l)):
                    if tagStart <= l <= tagStop:
                        tns.append(l)
        tns.sort()
        return tns

    