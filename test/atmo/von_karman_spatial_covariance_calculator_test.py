'''
Created on 3 mag 2022

@author: giuliacarla
'''


import unittest
import numpy as np
from arte.atmo.cn2_profile import Cn2Profile
from arte.types.guide_source import GuideSource
from arte.types.aperture import CircularOpticalAperture
from arte.atmo.von_karman_covariance_calculator import \
    VonKarmanSpatioTemporalCovariance
from arte.atmo.von_karman_spatial_covariance_calculator import \
    VonKarmanSpatialCovariance


class VonKarmanSpatialCovarianceCalculatorTest(unittest.TestCase):


    def testVonKarmanSpatialCovarianceVsVonKarmanSpatioTemporalCovariance(self):
        j1 = [2, 3, 4, 5, 6, 8, 9, 10]
        j2 = j1
        D = 40
        r0 = 0.16
        L0 = 25
        L0norm = L0 / D
        conversion_factor = (D / r0)**(5/3) * 4 * np.pi**2
        cn2 = Cn2Profile.from_r0s(
            [r0], [L0], [10e3], [10], [0])
        source = GuideSource((0, 0), np.inf)
        aperture = CircularOpticalAperture(D / 2, [0, 0, 0])
        spatial_freqs = np.logspace(-3, 3, 100)
        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        cov1 = vk.getZernikeCovariance(j1, j2).value
        
        cov2 = np.array([[
                VonKarmanSpatialCovariance(j, k, L0norm).get_covariance() 
            for k in j2] 
                for j in j1]) * conversion_factor
        np.testing.assert_allclose(cov1, cov2, atol=1e-1)