#!/usr/bin/env python
import unittest
import numpy as np

from arte.time_series.indexer import Indexer, ModeIndexer
from arte.time_series.indexer import RowColIndexer, DefaultIndexer


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

    def test_rowcol_indexer(self):

        indexer = RowColIndexer()
        rows, cols = indexer.rowcol(rows=3, col_from=0, col_to=10)
        assert rows == 3
        assert cols == slice(0, 10, None)

    def test_default_indexer_all(self):
        indexer = DefaultIndexer()
        ret = indexer.elements()
        assert ret == slice(None, None, None)

    def test_default_indexer_single_element(self):
        indexer = DefaultIndexer()
        ret = indexer.elements(22)
        assert ret == 22

    def test_default_indexer_list(self):
        indexer = DefaultIndexer()
        ret = indexer.elements([2,3])
        assert ret == [2,3]

    def test_default_indexer_from(self):
        indexer = DefaultIndexer()
        ret = indexer.elements(from_element=10)
        assert ret == slice(10, None, None)

    def test_default_indexer_to(self):
        indexer = DefaultIndexer()
        ret = indexer.elements(to_element=10)
        assert ret == slice(None, 10, None)

    def test_default_indexer_fromto(self):
        indexer = DefaultIndexer()
        ret = indexer.elements(from_element=10, to_element=20)
        assert ret == slice(10, 20, None)

if __name__ == "__main__":
    unittest.main()
