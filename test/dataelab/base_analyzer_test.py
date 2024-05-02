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

