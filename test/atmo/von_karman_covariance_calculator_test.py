#!/usr/bin/env python
import unittest
import numpy as np
from astropy import units as u
from arte.atmo.von_karman_covariance_calculator import \
    VonKarmanSpatioTemporalCovariance
from arte.atmo.cn2_profile import Cn2Profile
from arte.types.guide_source import GuideSource
from arte.types.aperture import CircularOpticalAperture
# from test.test_helper import setUpLogger
# import logging
import sys


class VonKarmanCovarianceCalculatorTest(unittest.TestCase):

    #     def setUp(self):
    #         setUpLogger(logging.DEBUG)
    #         pass

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

    def testZernikeCPSD(self):
        cn2 = Cn2Profile.from_r0s([0.16], [25], [10e3], [10], [-20])
        rho1, theta1 = (0, 0)
        rho2, theta2 = (50, 30)
        source1 = GuideSource((rho1, theta1), np.inf)
        source2 = GuideSource((rho2, theta2), 100e3)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)

        modoJ = 2
        modoK = 5
        temp_freqs = [0.05, 130, 250]

        vk_cov = VonKarmanSpatioTemporalCovariance(
            source1, source2, aperture, aperture, cn2, spatial_freqs)
        got = vk_cov.getZernikeCPSD(modoJ, modoK, temp_freqs)

        want = np.array([
            -2.753219e+00 + 1.446667e+00j,
            5.859243e-13 - 6.117817e-13j,
            -4.741206e-15 + 4.089497e-15j])

        np.testing.assert_allclose(want, got.value, rtol=1e-3)
        self.assertEqual(got.unit, u.rad ** 2 / u.Hz)

    def testPhaseCPSD(self):
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
        want = np.array([
            5.532213e+01 + 2.710243e+00j,
            7.927701e-12 - 2.069043e-10j,
            -1.142719e-10 + 3.395629e-11j])
        np.testing.assert_allclose(want, got, rtol=1e-3)

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

        vkCov = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        vkCpsd = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        cpsdMatrix = 2 * np.real(vkCpsd.getZernikeCPSD(j, k, temporal_freqs
                                                       ).value)
        covarianceMatrix = vkCov.getZernikeCovariance(j, k).value
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

    def testZernikeCPSDvsZernikeGeneralCPSD(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16, 3.14], [25, 12], [10e3, 200], [10, 8], [-20, 23])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)

        j = [2, 3]
        k = [2, 3, 4]

        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        cpsd = vk.getZernikeCPSD(j, k, temporal_freqs)
        cov_from_cpsd = np.trapz(2 * np.real(cpsd), temporal_freqs).value

        general_cpsd = vk.getGeneralZernikeCPSD(j, k, temporal_freqs)
        cov_from_general_cpsd = np.trapz(general_cpsd, temporal_freqs).value

        np.testing.assert_allclose(
            cov_from_cpsd, cov_from_general_cpsd, atol=1e-14)

    def testPhaseCPSDvsPhaseGeneralCPSD(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16, 3.14], [25, 12], [10e3, 200], [10, 8], [-20, 90])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 20
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-3, 3, 100)

        vk = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)

        cpsd = vk.getPhaseCPSD(temporal_freqs)
        general_cpsd = vk.getGeneralPhaseCPSD(temporal_freqs)

        np.testing.assert_allclose(
            2 * cpsd.real, general_cpsd, atol=1e-14)

    def testModifySourceAndAperture(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16, 3.14], [25, 12], [10e3, 200], [10, 8], [-20, 23])
        source1 = GuideSource((20, 0), np.inf)
        source2 = GuideSource((42, 0), np.inf)
        aperture1 = CircularOpticalAperture(5, [0, 0, 0])
        aperture2 = CircularOpticalAperture(5, [1, 0, 0])
        spatial_freqs = np.logspace(-3, 3, 10)
        temporal_freqs = np.logspace(-4, 4, 11)

        j = [2, 3]
        k = [2, 4]

        vk = VonKarmanSpatioTemporalCovariance(
            source1, source1, aperture1, aperture1, cn2, spatial_freqs)
        cpsd1 = vk.getGeneralZernikeCPSD(j, k, temporal_freqs).value

        vk.setSource1(source2)
        vk.setAperture2(aperture2)
        cpsd2 = vk.getGeneralZernikeCPSD(j, k, temporal_freqs).value
        self.assertFalse(np.allclose(
            cpsd1, cpsd2, atol=1e-14))

    @unittest.skipIf('cupy' not in sys.modules,
                     "Can't test code with cupy")
    def testZernikeCPSDOnGPU(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16, 3.14], [25, 12], [10e3, 200], [10, 8], [-20, 23])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)

        j = [2, 3]
        k = [2, 3, 4]

        vk_np = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        cpsd_np = vk_np.getZernikeCPSD(j, k, temporal_freqs)

        vk_cp = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        vk_cp.useGPU()
        cpsd_cp = vk_cp.getZernikeCPSD(j, k, temporal_freqs)

        np.testing.assert_allclose(
            cpsd_np, cpsd_cp, atol=1e-14)

    @unittest.skipIf('cupy' not in sys.modules,
                     "Can't test code with cupy")
    def testGeneralZernikeCPSDOnGPU(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16, 3.14], [25, 12], [10e3, 200], [10, 8], [-20, 23])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)

        j = [2, 3]
        k = [2, 3, 4]

        vk_np = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        cpsd_np = vk_np.getGeneralZernikeCPSD(j, k, temporal_freqs)

        vk_cp = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        vk_cp.useGPU()
        cpsd_cp = vk_cp.getGeneralZernikeCPSD(j, k, temporal_freqs)

        np.testing.assert_allclose(
            cpsd_np, cpsd_cp, atol=1e-14)

    @unittest.skipIf('cupy' not in sys.modules,
                     "Can't test code with cupy")
    def testPhaseCPSDOnGPU(self):
        cn2 = Cn2Profile.from_r0s(
            [0.16, 3.14], [25, 12], [10e3, 200], [10, 8], [-20, 23])
        rho, theta = (0, 0)
        source = GuideSource((rho, theta), np.inf)
        radius = 5
        center = [0, 0, 0]
        aperture = CircularOpticalAperture(radius, center)
        spatial_freqs = np.logspace(-3, 3, 100)
        temporal_freqs = np.logspace(-4, 4, 100)

        vk_np = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        cpsd_np = vk_np.getPhaseCPSD(temporal_freqs)

        vk_cp = VonKarmanSpatioTemporalCovariance(
            source, source, aperture, aperture, cn2, spatial_freqs)
        vk_cp.useGPU()
        cpsd_cp = vk_cp.getPhaseCPSD(temporal_freqs)

        np.testing.assert_allclose(
            cpsd_np, cpsd_cp, atol=1e-14)
        

if __name__ == "__main__":
    unittest.main()
