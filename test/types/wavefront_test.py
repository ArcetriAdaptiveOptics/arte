#!/usr/bin/env python
import unittest
import numpy as np
from arte.types.wavefront import Wavefront


class WavefrontTest(unittest.TestCase):

    def setUp(self):
        self._n_wf = 10000

    def _createZ(self):
        self._z = Wavefront(np.arange(self._n_wf, dtype=np.float32), 42)

    def testNumpy(self):
        c1 = np.arange(self._n_wf, dtype=np.float32)
        z = Wavefront.fromNumpyArray(c1)
        c2 = z.toNumpyArray()
        self.assertTrue(np.array_equal(c1, c2))

    def testComparison(self):
        coeffs = np.arange(self._n_wf, dtype=np.float32)
        counter = 42
        z1 = Wavefront.fromNumpyArray(coeffs, counter)
        z2 = Wavefront.fromNumpyArray(coeffs.copy(), counter)
        self.assertEqual(z1, z2)


if __name__ == "__main__":
    unittest.main()
