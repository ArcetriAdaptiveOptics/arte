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
#         height = 0
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-2, 2, 100)

        vk_cov = VonKarmannSpatioTemporalCovariance(
            cn2, source, source, aperture, aperture, spatial_freqs)

        self.assertEqual(0, vk_cov.getZernikeCovariance(2, 3))

    def testLayerProjectedSeparationOnXYPlane(self):
        cn2 = EsoEltProfiles.Median()
        source1 = GuideSource((0, 0), np.inf)
        source2 = GuideSource((50, 30), 100e3)
        aperture = CircularOpticalAperture(10, [0, 0, 0])
        spatial_freqs = np.logspace(-2, 2, 100)
        vk_cov = VonKarmannSpatioTemporalCovariance(
            cn2, source1, source2, aperture, aperture, spatial_freqs)
        s_l = np.array([
            vk_cov.layerProjectedAperturesSeparation(n)[2]
            for n in range(vk_cov._layersAlt.shape[0])])
        self.assertEqual(np.zeros(vk_cov._layersAlt.shape[0]).all(), s_l.all())


if __name__ == "__main__":
    unittest.main()
