# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import astropy.units as u
from pathlib import Path
from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.base_file_walker import AbstractFileNameWalker


class TestFileWalker(AbstractFileNameWalker):

    def snapshots_dir(self, tag):
        return Path(__file__).parents[0] / 'data' / 'snapshots'

    def snapshot_dir(self, tag):
        return self.snapshots_dir() / tag[:8] / tag


class TestAnalyzer(BaseAnalyzer):
    def _get_file_walker(self):
        return TestFileWalker()


class BaseAnalyzerTest(unittest.TestCase):
    def test_creation(self):
        ee = TestAnalyzer.get('20240404_024500')
        assert ee.snapshot_tag() == '20240404_024500'

