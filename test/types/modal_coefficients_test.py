#!/usr/bin/env python
import unittest
import numpy as np
from arte.types.modal_coefficients import ModalCoefficients

__version__ = "$$"

class ModalCoefficientsTest(unittest.TestCase):

    def setUp(self):
        self._nz = 21

    def _createZ(self):
        self._z = ModalCoefficients(np.arange(self._nz, dtype=np.float32), 42)

    def testNumpy(self):
        c1 = np.arange(self._nz, dtype=np.float32)
        z = ModalCoefficients.fromNumpyArray(c1)
        c2 = z.toNumpyArray()
        self.assertTrue(np.array_equal(c1, c2))

    def testComparison(self):
        coeffs1 = np.arange(self._nz, dtype=np.float32)
        coeffs2 = coeffs1.copy()
        counter = 42
        z1 = ModalCoefficients.fromNumpyArray(coeffs1, counter)
        z2 = ModalCoefficients.fromNumpyArray(coeffs2, counter)
        self.assertEqual(z1, z2)
        z3 = ModalCoefficients.fromNumpyArray(coeffs1 * 2, counter)
        self.assertNotEqual(z1, z3)

    def testReturnDictionary(self):
        coeffs = np.arange(3) * 0.1
        counter = 12
        fm = 2
        z = ModalCoefficients.fromNumpyArray(coeffs, counter, first_mode=fm)
        d = z.toDictionary()
        self.assertTrue(np.array_equal(list(d.keys()), fm+np.array([0, 1, 2])))
        self.assertTrue(np.array_equal(list(d.values()), np.array([0, 0.1, 0.2])))

    def testReturnLength(self):
        self._createZ()
        self.assertEqual(self._z.numberOfModes(), self._nz)

    def testGetZ(self):
        self._createZ()
        wantZernIndexes = [2, 4, 20]
        shouldGet = (np.array(wantZernIndexes)).astype(np.float32)
        didGet = self._z.getM(wantZernIndexes)
        self.assertTrue(
            np.array_equal(shouldGet, didGet), "wanted %s got %s" % (shouldGet, didGet)
        )


if __name__ == "__main__":
    unittest.main()
