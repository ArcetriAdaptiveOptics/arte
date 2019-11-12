#!/usr/bin/env python
import unittest
import numpy as np
from apposto.atmo.von_karmann_covariance_calculator import \
    VonKarmannSpatioTemporalCovariance
from apposto.atmo.cn2_profile import Cn2Profile
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
            0.1, [1.0], [25], [1], [10], [0])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), 1e10)
        radius = 10
        center = [0, 0]
        height = 0
        aperture = CircularOpticalAperture(radius, center, height)
        spatial_freqs = np.logspace(0.1, 100, 100)

        vk_cov = VonKarmannSpatioTemporalCovariance(
            cn2, source, source, aperture, aperture, spatial_freqs)

        self.assertEqual(0, vk_cov.getZernikeCovariance(2, 3))


if __name__ == "__main__":
    unittest.main()
