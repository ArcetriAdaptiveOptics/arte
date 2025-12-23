# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import unittest
from pathlib import Path

from arte.utils.not_available import NotAvailable
from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.base_analyzer_set import BaseAnalyzerSet
from arte.dataelab.base_file_walker import AbstractFileNameWalker
from arte.dataelab.cache_on_disk import cache_on_disk, clear_cache

class TestFileWalker(AbstractFileNameWalker):

    def snapshots_dir(self):
        return Path(__file__).parents[0] / 'data' / 'snapshots'

    def snapshot_dir(self, tag):
        return self.snapshots_dir() / tag[:8] / tag

    def find_tag_between_dates(self, tag_start, tag_stop):
        dirlist = os.listdir(self.snapshots_dir())
        tags = []
        for dir in dirlist:
            lst = os.listdir(os.path.join(self.snapshots_dir(), dir))
            tags += list(filter(lambda x: x >=tag_start and x<tag_stop, lst))
        return tags

class DummyAnalyzer(BaseAnalyzer):
    def __init__(self, snapshot_tag, recalc=False):
        super().__init__(snapshot_tag, recalc=recalc)
        self._counter = 0
        if not os.path.exists(TestFileWalker().snapshot_dir(snapshot_tag)):
            NotAvailable.transformInNotAvailable(self)
            return

    @cache_on_disk
    def a_method(self):
        self._counter += 1
        return self._counter


class DummyAnalyzerSet(BaseAnalyzerSet):
    def __init__(self, first, last=None, recalc=False):
        super().__init__(first, last, recalc=recalc,
                         file_walker=TestFileWalker(), analyzer_type=DummyAnalyzer)


class BaseAnalyzerSetTest(unittest.TestCase):
    def test_creation_fromto(self):
        set = DummyAnalyzerSet('20240404_024500', '20240404_024501')
        assert len(set) == 1

    def test_creation_single(self):
        set = DummyAnalyzerSet('20240404_024500')
        assert len(set) == 1

    def test_creation_recalc(self):
        set = DummyAnalyzerSet('20240404_024500', recalc=True)
        assert len(set) == 1

    def test_creation_list(self):
        set = DummyAnalyzerSet(['20240404_024500'])
        assert len(set) == 1

    def test_creation_list_notfound(self):
        set = DummyAnalyzerSet(['20240404_024500', '20240404_0245002'])
        assert len(set) == 2

    def test_creation_invalid(self):
        set = DummyAnalyzerSet(['20240404_024500', '20240404_0245002'])
        set.remove_invalids()
        assert len(set) == 1

    def test_attribute(self):
        set = DummyAnalyzerSet('20240404_024500', '20240404_024501')
        assert len(set.snapshot_tag()) == 1

    def test_invalid_attribute(self):
        set = DummyAnalyzerSet('20240404_024500', '20240404_024501')
        with self.assertRaises(AttributeError):
            _ = set.foo()

    def test_get(self):
        set = DummyAnalyzerSet('20240404_024500')
        ee = DummyAnalyzer.get('20240404_024500')
        assert id(set['20240404_024500']) == id(ee)

    def test_for(self):
        set = DummyAnalyzerSet(['20240404_024500', '20240404_0245002'])
        for ee in set:
            _ = ee.snapshot_tag()

    def test_recalc_works(self):
        set = DummyAnalyzerSet(['20240404_024500'], recalc=True)
        a = set.a_method()
        b = set.a_method()
        set = DummyAnalyzerSet(['20240404_024500'], recalc=True)
        c = set.a_method()
        assert a == b
        for aa, cc, in zip(a,c):
            assert cc == aa + 1
