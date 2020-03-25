#!/usr/bin/env python
import unittest
import numpy as np
from apposto.types.slopes import Slopes


class SlopesTest(unittest.TestCase):

    def setUp(self):
        self._n_slopes = 1600
        self._mapx = np.ma.masked_array(
            np.arange(self._n_slopes, dtype=np.float32))
        self._mapy = np.ma.masked_array(
            np.arange(self._n_slopes, dtype=np.float32)) * 10
        self._slopes = Slopes(self._mapx, self._mapy)

    def testNumpy(self):
        slopes = Slopes.fromNumpyArray(self._mapx, self._mapy)
        x2, y2 = slopes.toNumpyArray()
        self.assertTrue(np.array_equal(self._mapx, x2))
        self.assertTrue(np.array_equal(self._mapy, y2))

    def testComparison(self):
        s1 = Slopes.fromNumpyArray(self._mapx, self._mapy)
        mapX2 = self._mapx.copy()
        mapY2 = self._mapy.copy()
        self.assertTrue(mapX2 is not self._mapx)
        self.assertTrue(mapY2 is not self._mapy)
        s2 = Slopes.fromNumpyArray(mapX2, mapY2)
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()
