#!/usr/bin/env python
import unittest
import numpy as np
from arte.atmo.von_karman_covariance_calculator import \
    VonKarmanSpatioTemporalCovariance
from arte.atmo.cn2_profile import Cn2Profile
from arte.types.guide_source import GuideSource
from arte.types.aperture import CircularOpticalAperture
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
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-2, 2, 100)

        vk_cov = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        wanted = 0
        got = vk_cov.getZernikeCovariance(2, 3).value

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

        vk_cov = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        got = vk_cov.getZernikeCovariance(modes_j, modes_k).value

        self.assertAlmostEqual(vk_cov.getZernikeCovariance(2, 3).value,
                               got[0, 0])
        self.assertAlmostEqual(vk_cov.getZernikeCovariance(3, 3).value,
                               got[1, 0])
        self.assertAlmostEqual(vk_cov.getZernikeCovariance(4, 4).value,
                               got[2, 1])
        self.assertTrue(got.shape == (3, 2))

#     def testLayerProjectedSeparationOnXYPlane(self):
#         cn2 = EsoEltProfiles.Median()
#         source1 = GuideSource((0, 0), np.inf)
#         source2 = GuideSource((50, 30), 100e3)
#         aperture = CircularOpticalAperture(10, [0, 0, 0])
#         spatial_freqs = np.logspace(-2, 2, 100)
#
#         vk_cov = VonKarmanSpatioTemporalCovariance(
#             source1, source2, aperture, aperture, cn2, spatial_freqs)
#
#         zCoordOfLayersSeparations = np.array([
#             vk_cov.layerProjectedAperturesSeparation(n)[2]
#             for n in range(cn2.number_of_layers())])
#
#         want = np.zeros(cn2.number_of_layers())
#
#         assert_allclose(want, zCoordOfLayersSeparations, atol=1e-14)

    def testZernikeCPSD(self):
        i = np.complex(0, 1)
        cn2 = Cn2Profile.from_r0s([0.16], [25], [10e3], [10], [-20])
        rho1, theta1 = (0, 0)
        rho2, theta2 = (50, 30)
        source1 = GuideSource((rho1, theta1), np.inf)
        source2 = GuideSource((rho2, theta2), 100e3)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)

        vk_cov = VonKarmanSpatioTemporalCovariance(
            source1, source2, aperture, aperture, cn2, spatial_freqs)

        modoJ = 2
        modoK = 5
        temp_freqs = [0.05, 130, 250]

        got = vk_cov.getZernikeCPSD(modoJ, modoK, temp_freqs).value

        want = np.array([-2.7537494192458656 + 1.4469454298642168 * i,
                         5.860371984070442e-13 - 6.118996011218766e-13 * i,
                         -4.7421202823319255e-15 + 4.0902856472182244e-15 * i])

        np.testing.assert_allclose(want, got)
# TODO: FIXME TEST SOMETHING

    def testPhaseCPSD(self):
        i = np.complex(0, 1)
        cn2 = Cn2Profile.from_r0s([0.16], [25], [10e3], [10], [-20])
        rho1, theta1 = (0, 0)
        rho2, theta2 = (50, 30)
        source1 = GuideSource((rho1, theta1), np.inf)
        source2 = GuideSource((rho2, theta2), 100e3)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 1000)

        vk_cov = VonKarmanSpatioTemporalCovariance(
            source1, source2, aperture, aperture, cn2, spatial_freqs)

        temp_freqs = [0.05, 130, 250]

        got = vk_cov.getPhaseCPSD(temp_freqs).value
        want = np.array([55.33279564624362 + 2.7107654578823674 * i,
                         7.929228930453956e-12 - 2.0694414093390505e-10 * i,
                         -1.1429392989780903e-10 + 3.3962839307091425e-11 * i])
        np.testing.assert_allclose(want, got)

    def testIntegrationOfPhaseCPSDWithPhaseCovariance(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16], [25], [10e3], [10], [-20])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)

        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        phaseCov = vk.getPhaseCovariance().value
        phaseCPSD = vk.getPhaseCPSD(temporal_freqs).value
        phaseCovFromCPSD = np.trapz(2 * np.real(phaseCPSD), temporal_freqs)
        np.testing.assert_allclose(phaseCovFromCPSD, phaseCov, rtol=0.01)

    def testIntegrationOfZernikeCPSDWithZernikeCovariance(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16], [25], [10e3], [10], [-20])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-4, 4, 100)
        temporal_freqs = np.logspace(-3, 3, 100)

        j = [2, 3]
        k = [2, 3]

        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        cpsdMatrix = 2 * np.real(vk.getZernikeCPSD(j, k, temporal_freqs).value)
        covarianceMatrix = vk.getZernikeCovariance(j, k).value
        covFromCPSD = np.trapz(cpsdMatrix, temporal_freqs)
        np.testing.assert_allclose(
            covFromCPSD, covarianceMatrix, rtol=0.01, atol=1e-3)

    def testZernikeCPSDMatrixShape(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16], [25], [10e3], [10], [-20])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)
        j = [2]
        k = [2, 3]

        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        cpsdMatrix = vk.getZernikeCPSD(j, k, temporal_freqs).value
        self.assertTrue(cpsdMatrix.shape == (len(j), len(k),
                                             temporal_freqs.shape[0]))

    def testCPSDvsGeneralCPSD(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16], [25], [10e3], [10], [-20])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)
        j = [2, 3]
        k = [2, 3]

        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        cpsd = vk.getZernikeCPSD(j, k, temporal_freqs)
        cov_from_cpsd = np.trapz(2 * np.real(cpsd), temporal_freqs).value

        general_cpsd = vk.getGeneralZernikeCPSD(j, k, temporal_freqs)
        cov_from_general_cpsd = np.trapz(general_cpsd, temporal_freqs).value

        np.testing.assert_allclose(cov_from_cpsd, cov_from_general_cpsd)


if __name__ == "__main__":
    unittest.main()
