#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import BaseMask, CircularMask
from arte.utils.modal_decomposer import ModalDecomposer
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class ModalDecomposerTest(unittest.TestCase):

    def testMeasureZernikeCoefficientsFromSlopes(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        modes_idxs = list(range(2, 6))
        dx = zg.getDerivativeXDict(modes_idxs)
        dy = zg.getDerivativeYDict(modes_idxs)
        mapX = 2.5 * dx[2] - 4 * dx[3] + 3 * dx[5]
        mapY = 2.5 * dy[2] - 4 * dy[3] + 3 * dy[5]
        slopes = Slopes(mapX, mapY)
        modalDecomposer = ModalDecomposer(5)
        zernike = modalDecomposer.measureZernikeCoefficientsFromSlopes(
            slopes, mask, mask)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    [zernike.getZ(j) for j in modes_idxs]),
                        "zernike decomposition is %s" % str(zernike))

    # def testZernikeRecIsCached(self):
    #     md = ModalDecomposer(11)
    #     diameter = 100
    #     foo = 'foo'
    #     n_modes = 10
    #     rec1 = md._synthZernikeRecFromSlopes(n_modes, diameter, foo)
    #     rec2 = md._synthZernikeRecFromSlopes(n_modes, diameter, foo)
    #     self.assertTrue(rec1 is rec2)
    #     rec3 = md._synthZernikeRecFromSlopes(1, diameter, foo)
    #     self.assertFalse(rec1 is rec3)

    def testRaiseOnWrongArguments(self):
        md = ModalDecomposer(3)
        slopes = np.ones(100)
        mask = np.zeros((8, 8))
        self.assertRaises(
            Exception,
            md.measureZernikeCoefficientsFromSlopes, slopes, mask)

    def testCanSpecifyNumberOfModes(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        modes_idxs = list(range(2, 6))
        dx = zg.getDerivativeXDict(modes_idxs)
        dy = zg.getDerivativeYDict(modes_idxs)
        mapX = 2.5 * dx[2] - 4 * dx[3] + 3 * dx[5]
        mapY = 2.5 * dy[2] - 4 * dy[3] + 3 * dy[5]
        slopes = Slopes(mapX, mapY)
        modalDecomposer = ModalDecomposer(8)
        zernike = modalDecomposer.measureZernikeCoefficientsFromSlopes(slopes, mask)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    [zernike.getZ(j) for j in modes_idxs]),
                        "zernike decomposition is %s" % str(zernike))
        zernike42 = modalDecomposer.measureZernikeCoefficientsFromSlopes(
            slopes, mask, nModes=42)
        self.assertTrue(42, len(zernike42.toNumpyArray()))
        self.assertTrue(np.allclose(zernike.toNumpyArray()[0:4],
                                    zernike42.toNumpyArray()[0:4]),
                        "zernike decomposition is %s vs %s" % 
                        (str(zernike42), str(zernike)))

    def testMaskCompressedVectorSizeAgree(self):
        md = ModalDecomposer(3)
        mapX = np.ma.masked_array(np.arange(40000).reshape((200, 200)))
        mapY = mapX.T
        slopes = Slopes(mapX, mapY)
        for maskRadius in np.arange(1, 90, 7):
            mask = CircularMask((200, 200), maskRadius, [101, 100])
            md.measureZernikeCoefficientsFromSlopes(slopes, mask)

    def testMeasureZernikeCoefficientsFromWavefront(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        modes_idxs = list(range(2, 6))
        zernModes = zg.getZernikeDict(modes_idxs)
        wavefront = 2.5 * zernModes[2] - 4 * zernModes[3] + 3 * zernModes[5]

        modalDecomposer = ModalDecomposer(5)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask, mask)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    [zernike.getZ(j) for j in modes_idxs]),
                        "zernike decomposition is %s" % str(zernike))
        
    def testMeasureZernikeCoefficientsFromWavefrontUsingDifferentMasks(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        mask1 = CircularMask((2 * radius, 2 * radius), radius)
        mask2 = CircularMask((2 * radius, 2 * radius), radius / 2)
        modes_idxs = list(range(2, 6))
        zernModes = zg.getZernikeDict(modes_idxs)
        wavefront = 2.5 * zernModes[2] - 4 * zernModes[3] + 3 * zernModes[5]

        modalDecomposer = ModalDecomposer(5)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask1, mask2)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    [zernike.getZ(j) for j in modes_idxs]),
                        "zernike decomposition is %s" % str(zernike))

    def testZernikeCoefficientsFromWfRejectsPiston(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        wavefront = 100.* zg.getZernike(1) + 1.* zg.getZernike(2)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        modalDecomposer = ModalDecomposer(4)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask, mask)
        self.assertAlmostEqual(1, zernike.getZ(2))
        self.assertAlmostEqual(0, zernike.getZ(1),
            msg="zernike decomposition is %s" % str(zernike))

    def testMaskWithSmallerSlopeMap(self):
        md = ModalDecomposer(3)
        slopeMask = CircularMask((200, 200), 90, [100, 100])
        slopeMask.mask()[100, 100] = True   # Mask out slopes inside the pupil
        mapX = np.ma.masked_array(np.arange(40000).reshape(
            (200, 200)), mask=slopeMask.mask())
        mapY = mapX.T
        slopes = Slopes(mapX, mapY)
        mask = CircularMask((200, 200), 90, [100, 100])
        md.measureZernikeCoefficientsFromSlopes(
            slopes, mask, BaseMask(slopeMask.mask()))


if __name__ == "__main__":
    unittest.main()
