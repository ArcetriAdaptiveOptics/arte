#!/usr/bin/env python

import unittest
import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask
from numpy.testing import assert_allclose


class TestZernikeGenerator(unittest.TestCase):

    def setUp(self):
        self._nPixels = 10
        self.generator = ZernikeGenerator(self._nPixels)

    def testPiston(self):
        piston = self.generator[1]
        self.assertEqual(piston.shape, (self._nPixels, self._nPixels))
        self.assertTrue(piston.mask[0, 0])
        self.assertAlmostEqual(
            piston[self._nPixels // 2, self._nPixels // 2], 1.)

    def testTilt(self):
        nPx = 128
        generator = ZernikeGenerator(nPx)
        tilt = generator[3]
        self.assertAlmostEqual(tilt[-1, nPx // 2], 2. * (1 - 1. / nPx))
        self.assertAlmostEqual(tilt[0, nPx // 2], -2. * (1 - 1. / nPx))

    def testTiltOdd(self):
        nPx = 127
        generator = ZernikeGenerator(nPx)
        tilt = generator[3]
        self.assertAlmostEqual(tilt[-1, nPx // 2], 2. * (1 - 1. / nPx))

    def testOddShape(self):
        nPx = 3
        generator = ZernikeGenerator(nPx)
        tip = generator[2]
        self.assertAlmostEqual(tip[1, 1], 0.)
        self.assertEqual(tip.shape, (nPx, nPx))

    def testDegree(self):
        m, n = ZernikeGenerator.degree(1)
        self.assertEqual(m, 0)
        self.assertEqual(n, 0)
        m, n = ZernikeGenerator.degree(2)
        self.assertEqual(m, 1)
        self.assertEqual(n, 1)
        m, n = ZernikeGenerator.degree(3)
        self.assertEqual(m, 1)
        self.assertEqual(n, 1)
        m, n = ZernikeGenerator.degree(4)
        self.assertEqual(m, 2)
        self.assertEqual(n, 0)
        m, n = self.generator.degree(2000000)
        self.assertEqual(m, 1999)
        self.assertEqual(n, 999)

    def testRmn(self):
        self.assertTrue(np.allclose(
            self.generator._rnm_jacobi(0, 0, np.array([0, 0.5, 1])),
            np.array([1, 1, 1])))
        self.assertTrue(np.allclose(
            self.generator._rnm_jacobi(1, 1, np.array([0, 0.5, 1])),
            np.array([0, 0.5, 1])))
        self.assertTrue(np.allclose(
            self.generator._rnm_jacobi(2, 0, np.array([0, 0.5, 1])),
            np.array([-1, -0.5, 1])))
        self.assertTrue(np.allclose(
            self.generator._rnm_jacobi(5, 1, np.array([0, 0.5, 1])),
            np.array([0, 0.3125, 1])))

    def testPolar(self):
        self.assertTrue(np.allclose(
            self.generator._polar(1, np.array([0, 0.5, 1]),
                                  np.array([0, 1, 5]) * np.pi / 4),
            np.array([1, 1, 1])))
        self.assertTrue(np.allclose(
            self.generator._polar(2, np.array([0, 0.5, 1]),
                                  np.array([0, 1, 5]) * np.pi / 3),
            np.array([0, 0.5, 1])))
        self.assertTrue(np.allclose(
            self.generator._polar(3, np.array([0, 0.5, 1]),
                                  np.array([0, 1, 5]) * np.pi / 3),
            np.array([0, 0.866025, -1.73205])))
        self.assertTrue(np.allclose(
            self.generator._polar(16, np.array([0, 0.5, 1]),
                                  np.array([0, 1, 5]) * np.pi / 3),
            np.array([0, 0.541266, 1.73205])))

    def testDerivativeX(self):
        gammaX = self.generator._derivativeCoeffX(30)
        self.assertAlmostEqual(gammaX[1, 0], 2)
        self.assertAlmostEqual(gammaX[15, 3], 6)
        self.assertAlmostEqual(gammaX[16, 4], 3 * np.sqrt(2))

    def testDerivativeX2(self):
        gammaX = self.generator._derivativeCoeffX(2)
        self.assertTrue(np.allclose(gammaX,
                                    np.array([[0., 0.], [2., 0.]])),
                        "%s" % str(gammaX))

    def testDerivativeY3(self):
        gammaY = self.generator._derivativeCoeffY(3)
        self.assertTrue(np.allclose(gammaY,
                                    np.array([[0., 0., 0.],
                                              [0., 0., 0.],
                                              [2., 0., 0.]])),
                        "%s" % str(gammaY))

    def testDerivativeY(self):
        gammaY = self.generator._derivativeCoeffY(30)
        self.assertAlmostEqual(gammaY[2, 0], 2)
        self.assertAlmostEqual(gammaY[16, 3], 6)
        self.assertAlmostEqual(gammaY[9, 4], -2 * np.sqrt(3))

    def testRmsIsOne(self):
        zg = ZernikeGenerator(512)
        self.assertAlmostEqual(1., np.std(zg.getZernike(4)), 4)

    def testGetItem(self):
        a = self.generator[2]
        b = self.generator.getZernike(2)
        self.assertTrue(np.allclose(a, b))

    def testGetZernikeDict(self):
        dd = self.generator.getZernikeDict([1, 2, 10])
        self.assertTrue(np.allclose([1, 2, 10], list(dd.keys())))
        self.assertTrue(np.allclose(dd[10], self.generator.getZernike(10)))

    def testGetDerivativeOfTip(self):
        got = self.generator.getDerivativeX(2)
        wanted = 2 * np.ones((self._nPixels, self._nPixels))
        self.assertTrue(np.allclose(wanted, got),
                        "got %s, wanted %s" % (str(got), str(wanted)))
        got = self.generator.getDerivativeY(2)
        wanted = 0. * np.ones((self._nPixels, self._nPixels))
        self.assertTrue(np.allclose(wanted, got),
                        "got %s, wanted %s" % (str(got), str(wanted)))

    def testGetDerivativeOfTilt(self):
        got = self.generator.getDerivativeX(3)
        wanted = 0. * np.ones((self._nPixels, self._nPixels))
        self.assertTrue(np.allclose(wanted, got),
                        "got %s, wanted %s" % (str(got), str(wanted)))
        got = self.generator.getDerivativeY(3)
        wanted = 2. * np.ones((self._nPixels, self._nPixels))
        self.assertTrue(np.allclose(wanted, got),
                        "got %s, wanted %s" % (str(got), str(wanted)))

    def test_radial_order(self):
        self.assertEqual(1, self.generator.radial_order(2))
        self.assertEqual(1, self.generator.radial_order(3))
        self.assertEqual(2, self.generator.radial_order(4))
        self.assertEqual(2, self.generator.radial_order(5))
        self.assertEqual(2, self.generator.radial_order(6))
        self.assertEqual(4, self.generator.radial_order(15))
        assert_allclose(
            np.array([2, 4]), self.generator.radial_order(np.array([4, 15])))
        assert_allclose(np.array([2, 4]), self.generator.radial_order([4, 15]))

#     def testGetDerivativeOfFocus(self):
#         got= self.generator.getDerivativeX(4)
#         wanted= np.(self._nPixels)
#         self.assertTrue(np.allclose(wanted, got),
#                         "got %s, wanted %s" % (str(got), str(wanted)))

    def testNonIntegerPupilDiameterHasRoundedUpShape(self):
        nPx = 2.5
        generator = ZernikeGenerator(nPx)
        tip = generator[2]
        self.assertEqual(tip.shape, (3, 3))

    def testNonIntegerPupilDiameterTilt(self):
        diameter = 2.5
        generator = ZernikeGenerator(diameter)
        tilt = generator[3]
        self.assertAlmostEqual(tilt[2, 1], 2. * 0.8)
        self.assertAlmostEqual(tilt[0, 1], -2. * 0.8)

    def testFromCircularMask(self):
        mask = CircularMask((3, 5), maskRadius=1.25)
        generator = ZernikeGenerator(mask)
        tilt = generator[3]

        self.assertEqual(tilt.shape, mask.shape())
        self.assertAlmostEqual(tilt[2, 2], 2. * 0.8)
        self.assertAlmostEqual(tilt[0, 2], -2. * 0.8)

    def testCartesianCoordinates(self):
        nPx = 3
        zg = ZernikeGenerator(nPx)
        x, y = zg.cartesian_coordinates()
        want = np.array([-2 / 3, 0, 2 / 3])
        np.testing.assert_allclose(x[0], want)
        want = np.array([-2 / 3, -2 / 3, -2 / 3])
        np.testing.assert_allclose(y[0], want)

    def testInvalidIndexRaisesException(self):
        nPx = 16
        zg = ZernikeGenerator(nPx)
        self.assertRaises(ValueError, zg.getZernike, 2.5)
        self.assertRaises(ValueError, zg.getZernike, -1)


    def testIntegerIndexWithStupidNumpyIntegers(self):
        nPx = 16
        zg = ZernikeGenerator(nPx)
        index_vector = np.arange(2,4)
        zg.getZernikeDict(index_vector)

    def testRmsOfMode2347(self):
        zg = ZernikeGenerator(256)
        self.assertAlmostEqual(1., np.std(zg.getZernike(2347)), delta=0.02)


if __name__ == "__main__":
    unittest.main()
