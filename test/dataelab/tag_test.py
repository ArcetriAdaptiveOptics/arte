# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest

from arte.dataelab.tag import Tag


class TagTest(unittest.TestCase):
    def test_date_in_seconds(self):
        tag = Tag('TEST_20240404_024500')
        seconds = tag.date_in_seconds()
        assert seconds == 1712198700.0