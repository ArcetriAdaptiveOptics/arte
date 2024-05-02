# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import glob
import unittest
from pathlib import Path

from arte.utils.not_available import NotAvailable
from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.base_analyzer_set import BaseAnalyzerSet
from arte.dataelab.base_file_walker import AbstractFileNameWalker


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


class TestAnalyzer(BaseAnalyzer):
    def __init__(self, snapshot_tag, recalc=False):
       if not os.path.exists(TestFileWalker().snapshot_dir(snapshot_tag)):
            NotAvailable.transformInNotAvailable(self)
            return


class TestAnalyzerSet(BaseAnalyzerSet):
    def _get_file_walker(self):
        return TestFileWalker()

    def _get_type(self):
        return TestAnalyzer


class BaseAnalyzerSetTest(unittest.TestCase):
    def test_creation_fromto(self):
        set = TestAnalyzerSet('20240404_024500', '20240404_024501')
        assert len(set) == 1

    def test_creation_single(self):
        set = TestAnalyzerSet('20240404_024500')
        assert len(set) == 1

    def test_creation_recalc(self):
        set = TestAnalyzerSet('20240404_024500', recalc=True)
        assert len(set) == 1

    def test_creation_list(self):
        set = TestAnalyzerSet(['20240404_024500'])
        assert len(set) == 1

    def test_creation_list_notfound(self):
        set = TestAnalyzerSet(['20240404_024500', '20240404_0245002'])
        assert len(set) == 1

    def test_creation_invalid(self):
        set = TestAnalyzerSet(['20240404_024500', '20240404_0245002'], skip_invalid=False)
        assert len(set) == 2

    def test_stub(self):
        set = TestAnalyzerSet('20240404_024500', '20240404_024501')
        assert len(set.tip_tilt_slopes.get_data()) == 1

    @unittest.skip('For some reason this fails.. investigate')
    def test_get(self):
        set = TestAnalyzerSet('20240404_024500')
        ee = TestAnalyzer.get('20240404_024500')
        assert id(set['20240404_024500']) == id(ee)

    def test_for(self):
        set = TestAnalyzerSet(['20240404_024500', '20240404_0245002'])
        for ee in set:
            _ = ee.snapshot_tag()
