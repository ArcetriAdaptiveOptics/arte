#!/usr/bin/env python
import unittest
import numpy as np

from arte.time_series.indexer import Indexer, ModeIndexer


class TestIndexer(unittest.TestCase):

    def test_all_modes(self):
        self.indexer = ModeIndexer(max_mode=100)
        ret = self.indexer.modes()
        self.assertTrue(
            np.allclose(ret, np.arange(0, 100)))

    def test_range_of_modes(self):
        self.indexer = ModeIndexer(max_mode=100)
        ret = self.indexer.modes(from_mode=42, to_mode=142)
        self.assertTrue(
            np.allclose(ret, np.arange(42, 142)))

    def test_vector_of_modes(self):
        self.indexer = ModeIndexer(max_mode=100)
        wantModes = np.array([1, 2, 13, 45])
        gotModes = self.indexer.modes(modes=wantModes)
        self.assertTrue(
            np.allclose(wantModes, gotModes))

    def test_xy_without_keyword(self):
        self.indexer = Indexer()
        ret = self.indexer.xy()
        self.assertEqual(ret, [0, 1])

    def test_xy_with_keyword(self):
        self.indexer = Indexer()
        ret = self.indexer.xy(axis='x')
        self.assertEqual(ret, [0])
        ret = self.indexer.xy(axis='y')
        self.assertEqual(ret, [1])
        ret = self.indexer.xy(axis=['x', 'y'])
        self.assertEqual(ret, [0, 1])
        ret = self.indexer.xy(coord='x')
        self.assertEqual(ret, [0])
        ret = self.indexer.xy(coord='y')
        self.assertEqual(ret, [1])
        ret = self.indexer.xy(coord=['x', 'y'])
        self.assertEqual(ret, [0, 1])

    def test_xy_with_wrong_keyword_raises(self):
        self.indexer = Indexer()
        self.assertRaises(
            Exception, lambda: self.indexer.xy(wrongKeyword='x'))


if __name__ == "__main__":
    unittest.main()
