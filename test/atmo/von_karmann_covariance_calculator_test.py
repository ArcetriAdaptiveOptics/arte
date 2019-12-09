#!/usr/bin/env python
import unittest
import numpy as np
from apposto.atmo.von_karmann_covariance_calculator import \
    VonKarmannSpatioTemporalCovariance
from apposto.atmo.cn2_profile import Cn2Profile, EsoEltProfiles
from apposto.types.guide_source import GuideSource
from apposto.types.aperture import CircularOpticalAperture
from test.test_helper import setUpLogger
import logging
from numpy.testing.nose_tools.utils import assert_allclose


class VonKarmanCovarianceCalculatorTest(unittest.TestCase):

    def setUp(self):
        setUpLogger(logging.DEBUG)
        pass

    def tearDown(self):
        pass

    def testCovarianceZernikeOnTheSameSourceAndAperture(self):
        cn2 = Cn2Profile.from_fractional_j(
            0.1, [1.0], [25], [0], [10], [0])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 10
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-2, 2, 100)

        vk_cov = VonKarmannSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        wanted = 0
        got = vk_cov.getZernikeCovariance(2, 3)

        self.assertAlmostEqual(wanted, got)

    def testCovarianceZernikeOnSeveralModes(self):
        cn2 = Cn2Profile.from_fractional_j(
            0.1, [1.0], [25], [0], [10], [0])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 10
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-2, 2, 100)

        modes_j = [2, 3, 4]
        modes_k = [3, 4]

        vk_cov = VonKarmannSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        got = vk_cov.getZernikeCovarianceMatrix(modes_j, modes_k)

        self.assertAlmostEqual(vk_cov.getZernikeCovariance(2, 3),
                               got[0, 0])
        self.assertAlmostEqual(vk_cov.getZernikeCovariance(3, 3),
                               got[1, 0])
        self.assertAlmostEqual(vk_cov.getZernikeCovariance(4, 4),
                               got[2, 1])
        self.assertTrue(got.shape == (3, 2))

    def testLayerProjectedSeparationOnXYPlane(self):
        cn2 = EsoEltProfiles.Median()
        source1 = GuideSource((0, 0), np.inf)
        source2 = GuideSource((50, 30), 100e3)
        aperture = CircularOpticalAperture(10, [0, 0, 0])
        spatial_freqs = np.logspace(-2, 2, 100)

        vk_cov = VonKarmannSpatioTemporalCovariance(
            source1, source2, aperture, aperture, cn2, spatial_freqs)

        zCoordOfLayersSeparations = np.array([
            vk_cov.layerProjectedAperturesSeparation(n)[2]
            for n in range(cn2.number_of_layers())])

        want = np.zeros(cn2.number_of_layers())

        assert_allclose(want, zCoordOfLayersSeparations, atol=1e-14)

    def testZernikeCPSD(self):
        cn2 = EsoEltProfiles.Median()
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 10
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-2, 2, 100)

        vk_cov = VonKarmannSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        nLayer = 0
        modoJ = 2
        modoK = 3
        temp_freqs = np.linspace(3.14, 42, 10)
        vk_cov.getZernikeCPSD(modoJ, modoK, nLayer, temp_freqs)
# TODO: FIXME TEST SOMETHING


if __name__ == "__main__":
    unittest.main()
