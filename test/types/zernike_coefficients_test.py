#!/usr/bin/env python
import unittest
import numpy as np
from arte.types.zernike_coefficients import ZernikeCoefficients


class ZernikeCoefficientsTest(unittest.TestCase):

    def setUp(self):
        self._nz = 21

    def _createZ(self):
        self._z = ZernikeCoefficients(np.arange(self._nz, dtype=np.float32), 42)

    def testNumpy(self):
        c1 = np.arange(self._nz, dtype=np.float32)
        z = ZernikeCoefficients.fromNumpyArray(c1)
        c2 = z.toNumpyArray()
        self.assertTrue(np.array_equal(c1, c2))

    def testComparison(self):
        coeffs1 = np.arange(self._nz, dtype=np.float32)
        coeffs2 = coeffs1.copy()
        counter = 42
        z1 = ZernikeCoefficients.fromNumpyArray(coeffs1, counter)
        z2 = ZernikeCoefficients.fromNumpyArray(coeffs2, counter)
        self.assertEqual(z1, z2)
        z3 = ZernikeCoefficients.fromNumpyArray(coeffs1 * 2, counter)
        self.assertNotEqual(z1, z3)

    def testReturnDictionary(self):
        coeffs = np.arange(3) * 0.1
        counter = 12
        z = ZernikeCoefficients.fromNumpyArray(coeffs, counter)
        d = z.toDictionary()
        self.assertTrue(np.array_equal(list(d.keys()), np.array(
            np.arange(3)+ZernikeCoefficients.FIRST_ZERNIKE_MODE)))
        self.assertTrue(np.array_equal(list(d.values()), np.array([0, 0.1, 0.2])))

    def testReturnLength(self):
        self._createZ()
        self.assertEqual(self._z.numberOfModes(), self._nz)

    def testGetZ(self):
        self._createZ()
        wantZernIndexes = [2, 4, 21]
        shouldGet = (np.array(wantZernIndexes) -
                     ZernikeCoefficients.FIRST_ZERNIKE_MODE).astype(np.float32)
        didGet = self._z.getZ(wantZernIndexes)
        self.assertTrue(
            np.array_equal(shouldGet, didGet), "wanted %s got %s" % (shouldGet, didGet)
        )

    def test_neg(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        zneg = -z1
        np.testing.assert_allclose(
            zneg.toNumpyArray(), -z1.toNumpyArray())

    def test_sum_easy(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 43)
        zsum = z1+z2
        np.testing.assert_allclose(
            zsum.toNumpyArray(), z1.toNumpyArray()+z2.toNumpyArray())

    def test_sum_scalar(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        z2 = 3.14
        zsum = z1+z2
        np.testing.assert_allclose(
            zsum.toNumpyArray(), z1.toNumpyArray()+z2)

    def test_sum_scalar_r(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        z2 = 3.14
        zsum = z2+z1
        np.testing.assert_allclose(
            zsum.toNumpyArray(), z1.toNumpyArray()+z2)

    def test_sum_not_supported_raises(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        z2 = 'sadf'

        def callit():
            return z1+z2
        self.assertRaises(TypeError, callit)

    def test_sum_in_place(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        zorig = z1*1
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 43)
        z1 += z2
        np.testing.assert_allclose(
            z1.toNumpyArray(), z2.toNumpyArray()+zorig.toNumpyArray())

    def test_difference_easy(self):
        z1 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 42)
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32)*3, 43)
        zsum = z2-z1
        np.testing.assert_allclose(
            zsum.toNumpyArray(), z2.toNumpyArray()-z1.toNumpyArray())

    def test_sum_different_size(self):
        z1 = ZernikeCoefficients(np.arange(10, dtype=np.float32), 42)
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 43)
        zsum = z1+z2
        wanted = np.zeros(21)
        wanted[:10] = np.arange(10)
        wanted += np.arange(21)
        np.testing.assert_allclose(
            zsum.toNumpyArray(), wanted)

    def test_sum_different_size_in_place(self):
        z1 = ZernikeCoefficients(np.arange(10, dtype=np.float32), 42)
        z2 = ZernikeCoefficients(np.arange(21, dtype=np.float32), 43)
        z1 += z2
        wanted = np.zeros(21)
        wanted[:10] = np.arange(10)
        wanted += np.arange(21)
        np.testing.assert_allclose(z1.toNumpyArray(), wanted)

    def test_scalar_multiplication(self):
        z1 = ZernikeCoefficients(np.arange(10, dtype=np.float32), 42)
        z2 = z1*3.14
        np.testing.assert_allclose(z2.toNumpyArray(), z1.toNumpyArray()*3.14)


if __name__ == "__main__":
    unittest.main()
