#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask
from arte.utils.karhunen_loeve_decomposer import KarhunenLoeveModalDecomposer
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes
from arte.types.modal_coefficients import ModalCoefficients
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.radial_basis_decomposer import RadialBasisModalDecomposer


class ModalDecomposerTest(unittest.TestCase):

    def setUp(self):
        self._center = (55, 60)
        self._radius = 32
        self._nPxX = 150
        self._nPxY = 101
        self._nModes = 100
        self._nCoord = self._nModes
        self._mask = CircularMask((self._nPxX, self._nPxY), self._radius, self._center)
        self._user_mask = CircularMask((self._nPxX, self._nPxY), 32, self._center)
        self._wavefront = Wavefront(np.zeros((self._nPxY, self._nPxX)))
        _mask = CircularMask((self._nPxY, self._nPxX), self._radius, self._center)

        y0, x0 = self._mask.center()
        cc = np.expand_dims((x0, y0), axis=(1, 2))
        Y, X = (
            np.mgrid[0.5 : self._nPxY + 0.5 : 1, 0.5 : self._nPxX + 0.5 : 1] - cc
        ) / self._mask.radius()
        r = np.sqrt(X**2 + Y**2)
        self._wavefront = Wavefront(r)
        self._user_mask = CircularMask((self._nPxX, self._nPxY), 30, self._center)

    def testZernikeModalDecomposer(self):
        self._base = "ZERNIKE"
        self._modal_decomposer = ModalDecomposer(self._nModes)
        c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront, self._mask, self._user_mask, 10, start_mode=1
        )
        cTemplate = [
            -1.30027833e-02,
            -3.22827224e-05,
            -1.06959015e-01,
            8.49964362e-03,
            1.51368177e-17,
            -7.18398687e-17,
            9.10631412e-02,
            2.74849775e-05,
            -1.24063393e-04,
            3.74452222e-08,
        ]
        cTemplate = [
            0.00875421,
            -0.00187142,
            -0.17916746,
            0.04178357,
            0.03762525,
            -0.02406837,
            0.00624326,
            -0.00212699,
            -0.00512877,
            -0.00418178,
        ]
        np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)
        tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront,
            self._mask,
            self._user_mask,
            10,
            start_mode=1,
            rtol=0.995,
        )
        np.testing.assert_allclose(self._modal_decomposer.getRank(), 3)

    def testRBFModalDecomposer(self):
        self._base = "TPS_RBF"
        self._coords = (
            (70, 60),
            (40, 40),
            (60, 60),
            (60, 80),
            (70, 60 + 2),
            (40 - 2, 40),
            (60 - 3, 60),
            (60, 80 + 3),
            (70 + 5, 60),
            (40, 40 + 5),
        )
        self._modal_decomposer = RadialBasisModalDecomposer(self._coords)

        c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront,
            self._base,
            self._mask,
            self._user_mask,
            coordinates_list=self._coords,
        )

        cTemplate = [
            -0.598514,
            0.578509,
            1.123119,
            0.561612,
            1.730237,
            -1.918513,
            -2.387599,
            -0.224815,
            -0.930808,
            1.880122,
        ]

        np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)

        tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront,
            self._base,
            self._mask,
            self._user_mask,
            coordinates_list=self._coords,
            atol=0.995,
        )

        np.testing.assert_allclose(self._modal_decomposer.getRank(), 7)

    def testKarhunenLoeveModalDecomposer(self):
        self._base = "KL"
        self._modal_decomposer = KarhunenLoeveModalDecomposer(10)
        c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront, self._mask, self._user_mask
        )
        # cTemplate = [-0.15911395, -0.00040384, -0.04565834, -0.0225641,   0.03527367, -0.00075877,
        #    -0.04821382,  0.00063496,  0.03775376, -0.03239231]
        cTemplate = [
            -0.17927546,
            0.00180251,
            -0.02950138,
            -0.02406837,
            0.03762525,
            -0.00418178,
            -0.00512877,
            -0.00218569,
            0.00051625,
            -0.04127313,
        ]
        np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)
        tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront, self._mask, self._user_mask, 10, rtol=0.995
        )
        np.testing.assert_allclose(self._modal_decomposer.getRank(), 3)

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
        self.assertTrue(
            np.allclose(np.array([2.5, -4, 0, 3.0]), zernike.toNumpyArray()[0:4]),
            "zernike decomposition is %s" % str(zernike),
        )

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
            Exception, md.measureZernikeCoefficientsFromSlopes, slopes, mask
        )

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
        self.assertTrue(
            np.allclose(np.array([2.5, -4, 0, 3.0]), zernike.toNumpyArray()[0:4]),
            "zernike decomposition is %s" % str(zernike),
        )
        zernike42 = modalDecomposer.measureZernikeCoefficientsFromSlopes(
            slopes, mask, 42
        )
        self.assertTrue(42, len(zernike42.toNumpyArray()))
        self.assertTrue(
            np.allclose(zernike.toNumpyArray()[0:4], zernike42.toNumpyArray()[0:4]),
            "zernike decomposition is %s vs %s" % (str(zernike42), str(zernike)),
        )

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
            Wavefront.fromNumpyArray(wavefront), mask, mask
        )
        self.assertTrue(
            np.allclose(np.array([2.5, -4, 0, 3.0]), zernike.toNumpyArray()[0:4]),
            "zernike decomposition is %s" % str(zernike),
        )

    def testMeasureZernikeCoefficientsFromWavefrontUsingDifferentMasks(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        mask1 = CircularMask((2 * radius, 2 * radius), radius)
        mask2 = CircularMask((2 * radius, 2 * radius), radius / 2)
        zernModes = zg.getZernikeDict(list(range(2, 6)))
        wavefront = 2.5 * zernModes[2] - 4 * zernModes[3] + 3 * zernModes[5]

        modalDecomposer = ModalDecomposer(5)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask1, mask2
        )
        self.assertTrue(
            np.allclose(np.array([2.5, -4, 0, 3.0]), zernike.toNumpyArray()[0:4]),
            "zernike decomposition is %s" % str(zernike),
        )

    def testZernikeCoefficientsFromWfRejectsPiston(self):
        radius = 4
        zg = ZernikeGenerator(2 * radius)
        wavefront = 100.0 * zg.getZernike(1) + 1.0 * zg.getZernike(2)
        mask = CircularMask((2 * radius, 2 * radius), radius)
        modalDecomposer = ModalDecomposer(4)
        zernike = modalDecomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront.fromNumpyArray(wavefront), mask, mask
        )
        self.assertAlmostEqual(1, zernike.getZ(2))
        self.assertAlmostEqual(
            0,
            zernike.toNumpyArray()[1:].sum(),
            msg="zernike decomposition is %s" % str(zernike),
        )

    def _generate_coordinates(self, center_x, center_y, radius, num_points):
        # Initialize list to hold coordinates
        coordinates = []

        # Determine the number of circles needed
        area_per_point = np.pi * (radius**2) / num_points
        radius_step = np.sqrt(area_per_point / np.pi)

        # Start with the center point
        current_radius = 0
        while current_radius < radius:
            # Determine the number of points on this circle
            circumference = 2 * np.pi * current_radius
            points_on_circle = int(circumference / np.sqrt(area_per_point))

            # Generate points on this circle
            for i in range(points_on_circle):
                angle = 2 * np.pi * i / points_on_circle
                x = center_x + current_radius * np.cos(angle)
                y = center_y + current_radius * np.sin(angle)
                coordinates.append((x, y))

            # Move to the next circle
            current_radius += radius_step

        return coordinates

    def testMeasureZernikeCoefficientsFromWavefront(self):
        self._modal_decomposer = ModalDecomposer(self._nModes)
        modal_coefficients = (
            self._modal_decomposer.measureModalCoefficientsFromWavefront(
                self._wavefront, self._mask, self._user_mask, self._nModes, start_mode=1
            )
        )
        self.assertIsInstance(modal_coefficients, ModalCoefficients)
        self.assertEqual(modal_coefficients.numberOfModes(), self._nModes)
        self.assertEqual(modal_coefficients.counter(), 0)

    def testMeasureKLCoefficientsFromWavefront(self):
        self._base = "KL"
        self._modal_decomposer = KarhunenLoeveModalDecomposer(10)
        modal_coefficients = (
            self._modal_decomposer.measureModalCoefficientsFromWavefront(
                self._wavefront,
                self._mask,
                self._user_mask,
            )
        )
        self.assertIsInstance(modal_coefficients, ModalCoefficients)
        self.assertEqual(modal_coefficients.numberOfModes(), 10)
        self.assertEqual(modal_coefficients.counter(), 0)

    def testMeasureTPSRBFcoefficientsFromWavefront(self):
        self._coords = (
            (70, 60),
            (40, 40),
            (60, 60),
            (60, 80),
            (70, 60 + 2),
            (40 - 2, 40),
            (60 - 3, 60),
            (60, 80 + 3),
            (70 + 5, 60),
            (40, 40 + 5),
        )
        self._modal_decomposer = RadialBasisModalDecomposer(self._coords)
        modal_coefficients = (
            self._modal_decomposer.measureModalCoefficientsFromWavefront(
                self._wavefront,
                "TPS_RBF",
                self._mask,
                self._user_mask,
            )
        )
        self.assertIsInstance(modal_coefficients, ModalCoefficients)
        self.assertEqual(modal_coefficients.numberOfModes(), 10)
        self.assertEqual(modal_coefficients.counter(), 0)


if __name__ == "__main__":
    unittest.main()
