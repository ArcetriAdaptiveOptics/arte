#!/usr/bin/env python
import unittest
import numpy as np
from apposto.utils.zernike_generator import ZernikeGenerator
from apposto.types.mask import CircularMask
from apposto.utils.modal_decomposer import ModalDecomposer
from apposto.types.wavefront import Wavefront
from apposto.types.slopes import Slopes


class ModalDecomposerTest(unittest.TestCase):

    def testMeasureZernikeCoefficientsFromSlopes(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        dx = zg.getDerivativeXDict(list(range(2, 6)))
        dy = zg.getDerivativeYDict(list(range(2, 6)))
        mapX = 2.5 * dx[2] - 4 * dx[3] + 3 * dx[5]
        mapY = 2.5 * dy[2] - 4 * dy[3] + 3 * dy[5]
        slopes = Slopes(mapX, mapY)
        modalDecomposer = ModalDecomposer(5)
        zernike = modalDecomposer.measureZernikeCoefficientsFromSlopes(slopes, mask)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    zernike.toNumpyArray()[0:4]),
                        "zernike decomposition is %s" % str(zernike))

    def testZernikeRecIsCached(self):
        md = ModalDecomposer(11)
        diameter = 100
        n_modes = 10
        rec1 = md._synthZernikeRecFromSlopes(n_modes, diameter)
        rec2 = md._synthZernikeRecFromSlopes(n_modes, diameter)
        self.assertTrue(rec1 is rec2)
        rec3 = md._synthZernikeRecFromSlopes(1, diameter)
        self.assertFalse(rec1 is rec3)

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
        dx = zg.getDerivativeXDict(list(range(2, 6)))
        dy = zg.getDerivativeYDict(list(range(2, 6)))
        mapX = 2.5 * dx[2] - 4 * dx[3] + 3 * dx[5]
        mapY = 2.5 * dy[2] - 4 * dy[3] + 3 * dy[5]
        slopes = Slopes(mapX, mapY)
        modalDecomposer = ModalDecomposer(8)
        zernike = modalDecomposer.measureZernikeCoefficientsFromSlopes(slopes, mask)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    zernike.toNumpyArray()[0:4]),
                        "zernike decomposition is %s" % str(zernike))
        zernike42 = modalDecomposer.measureZernikeCoefficientsFromSlopes(slopes, mask, 42)
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
        zernModes = zg.getZernikeDict(list(range(2, 6)))
        wavefront = 2.5 * zernModes[2] - 4 * zernModes[3] + 3 * zernModes[5]

        modalDecomposer = ModalDecomposer(5)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask)
        self.assertTrue(np.allclose(np.array([2.5, -4, 0, 3.0]),
                                    zernike.toNumpyArray()[0:4]),
                        "zernike decomposition is %s" % str(zernike))

    def testZernikeCoefficientsFromWfRejectsPiston(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        wavefront = 100.* zg.getZernike(1) + 1.* zg.getZernike(2)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        modalDecomposer = ModalDecomposer(4)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask)
        self.assertAlmostEqual(1, zernike.getZ(2))
        self.assertAlmostEqual(
            0, zernike.toNumpyArray()[1:].sum(),
            msg="zernike decomposition is %s" % str(zernike))


if __name__ == "__main__":
    unittest.main()
