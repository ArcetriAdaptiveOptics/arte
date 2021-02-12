# -*- coding: utf-8 -*-

import unittest
from collections import abc
from arte.utils.circular_range import CircularRange

class CircularRangeTest(unittest.TestCase):

    def test_type(self):
        assert isinstance(CircularRange(1,2,3), abc.Iterable)

    def test_nowrap(self):
        assert list(CircularRange(1,3,5)) == [1,2]

    def test_wrap(self):
        assert list(CircularRange(3,1,5)) == [3,4,0]


if __name__ == "__main__":
    unittest.main()

