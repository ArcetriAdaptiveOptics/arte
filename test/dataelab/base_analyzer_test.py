# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest

from arte.dataelab.base_analyzer import BaseAnalyzer


class TestAnalyzer(BaseAnalyzer):
    pass


class BaseAnalyzerTest(unittest.TestCase):
    def test_creation(self):
        ee = TestAnalyzer.get('20240404_024500')
        assert ee.snapshot_tag() == '20240404_024500'

    def test_get_cache(self):
        ee1 = TestAnalyzer.get('20240404_024500')
        ee2 = TestAnalyzer.get('20240404_024500')
        assert id(ee1) == id(ee2)

    def test_recalc(self):
        _ = TestAnalyzer.get('20240404_024500', recalc=True)

    def test_date_in_seconds(self):
        seconds = TestAnalyzer.get('20240404_024500').date_in_seconds()
        assert seconds == 1712198700.0

    def test_info(self):
        info = TestAnalyzer.get('20240404_024500').info()
        assert 'snapshot_tag' in info
        assert info['snapshot_tag'] == '20240404_024500'
