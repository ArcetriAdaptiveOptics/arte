#!/usr/bin/env python
import unittest
import numpy as np
from apposto.atmo.cn2_profile import Cn2Profile, \
    MaoryProfiles, EsoEltProfiles
from apposto.utils.constants import Constants


class Cn2ProfileTest(unittest.TestCase):

    def testSingleLayer(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        self.assertEqual([100], pr.layers_distance().value)
        self.assertEqual(0.1, pr.r0().value)
        pr = Cn2Profile.from_fractional_j(
            0.12, [1.0], [30], [100], [10], [0.3])
        self.assertEqual(0.12, pr.r0().value)

    def testScalarAsArguments(self):
        pr = Cn2Profile.from_r0s(0.1, 30, 100, 10, 0.3)
        self.assertEqual(100, pr.layers_distance().value)
        self.assertEqual(0.1, pr.r0().value)
        pr = Cn2Profile.from_fractional_j(
            0.12, 1.0, 30, 100, 10, 0.3)
        self.assertEqual(0.12, pr.r0().value)

    def testSeeingAndR0ScaleWithWavelengths(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        pr.set_wavelength(500e-9)
        r0At500 = 0.1
        seeingAt500 = 0.98 * 500e-9 / r0At500 * Constants.RAD2ARCSEC
        self.assertEqual(r0At500, pr.r0().value)
        self.assertEqual(seeingAt500, pr.seeing().value)

        pr.set_wavelength(2200e-9)
        self.assertEqual(r0At500 * (2200 / 500) ** (6. / 5), pr.r0().value)
        self.assertEqual(seeingAt500 * (2200 / 500) ** (-1. / 5),
                         pr.seeing().value)

    def testSetAndGetAirmassAndZenithAngle(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        pr.set_zenith_angle(0)
        self.assertAlmostEqual(0, pr.zenith_angle().value)
        self.assertAlmostEqual(1, pr.airmass())
        pr.set_zenith_angle(60)
        self.assertAlmostEqual(60, pr.zenith_angle().value)
        self.assertAlmostEqual(2, pr.airmass())

    def testBuildFromFractionalJ(self):
        pr = Cn2Profile.from_fractional_j(0.1, 1.0, 30, 100, 10, 0.3)
        self.assertAlmostEqual(0.1, pr.r0().value)
        pr = Cn2Profile.from_fractional_j(
            0.13, [0.1, 0.9], [30, 42], [0, 100], [3.14, 10], [0, 0.3])
        self.assertAlmostEqual(0.13, pr.r0().value)

        dontSumToOne = np.array([0.1, 42])
        self.assertRaises(Exception, Cn2Profile.from_fractional_j,
                          42, dontSumToOne, 30, 100, 10, 0)

    def testR0ScaleWithAirmass(self):
        pr = Cn2Profile.from_fractional_j(0.1, 1.0, 30, 100, 10, 0.3)
        pr.set_zenith_angle(0)
        self.assertAlmostEqual(0.1, pr.r0().value)
        pr.set_zenith_angle(60)
        self.assertAlmostEqual(0.1 * 2 ** (-3. / 5), pr.r0().value)
        pr.set_zenith_angle(45)
        self.assertAlmostEqual(0.1 * np.sqrt(2) ** (-3. / 5), pr.r0().value)

    def testTheta0AtGroundIsInfinite(self):
        pr = Cn2Profile.from_fractional_j(0.1, 1.0, 30, 0, 10, 0.3)
        pr.set_zenith_angle(42)
        self.assertEqual(np.inf, pr.theta0())

    def testTheta0OfSingleLayer(self):
        zenAngle = 23.5
        airmass = Cn2Profile.zenith_angle_to_airmass(zenAngle)
        r0 = 0.13
        h = 1000.
        pr = Cn2Profile.from_fractional_j(r0, 1.0, 30, h, 10, 0.3)
        pr.set_zenith_angle(zenAngle)
        pr.set_wavelength(1.0e-6)
        wanted = (2.914 / 0.422727) ** (-3. / 5) * pr.r0().value / (
            h * airmass) * Constants.RAD2ARCSEC
        self.assertAlmostEqual(wanted, pr.theta0().value)

    def testAirmass(self):
        self.assertAlmostEqual(1, Cn2Profile.zenith_angle_to_airmass(0))
        self.assertAlmostEqual(2, Cn2Profile.zenith_angle_to_airmass(60))
        self.assertAlmostEqual(np.sqrt(2),
                               Cn2Profile.zenith_angle_to_airmass(45))

    def testMaoryPercentileProfiles(self):
        pr = MaoryProfiles.P10()
        self.assertEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.48, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2.7, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(5.18, pr.wind_speed()[0].value, delta=0.1)
        self.assertAlmostEqual(0, pr.wind_direction()[0].value)

        pr = MaoryProfiles.P25()
        self.assertEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.55, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2.2, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(5.3, pr.wind_speed()[0].value, delta=0.1)

        pr = MaoryProfiles.P50()
        self.assertEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.66, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.74, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(5.7, pr.wind_speed()[0].value, delta=0.1)

        pr = MaoryProfiles.P75()
        self.assertEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.78, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.32, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(6.2, pr.wind_speed()[0].value, delta=0.1)

        pr = MaoryProfiles.P90()
        self.assertEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.92, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.04, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(6.9, pr.wind_speed()[0].value, delta=0.1)

    def testTau0OfSingleLayer(self):
        zenAngle = 10.
        r0 = 0.13
        h = 1234.
        wspd = 10.
        pr = Cn2Profile.from_fractional_j(r0, 1.0, 30, h, wspd, 0.3)
        pr.set_zenith_angle(zenAngle)
        pr.set_wavelength(1.0e-6)
        wanted = (2.914 / 0.422727) ** (-3. / 5) * pr.r0().value / (
            wspd)
        self.assertAlmostEqual(wanted, pr.tau0().value)

    def testEsoProfiles(self):
        pr = EsoEltProfiles.Median()
        self.assertEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.644, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(5.786, pr.wind_speed()[0].value, delta=0.01)
        self.assertAlmostEqual(0.00535, pr.tau0().value, delta=0.0001)
        pr = EsoEltProfiles.Q1()
        self.assertAlmostEqual(0.234, pr.r0().value, delta=0.001)
        self.assertAlmostEqual(0.00808, pr.tau0().value, delta=0.00001)
        pr = EsoEltProfiles.Q2()
        self.assertAlmostEqual(0.178, pr.r0().value, delta=0.001)
        self.assertAlmostEqual(0.00612, pr.tau0().value, delta=0.00001)
        pr = EsoEltProfiles.Q3()
        self.assertAlmostEqual(0.139, pr.r0().value, delta=0.001)
        self.assertAlmostEqual(0.00478, pr.tau0().value, delta=0.00001)
        pr = EsoEltProfiles.Q4()
        self.assertAlmostEqual(0.097, pr.r0().value, delta=0.001)
        self.assertAlmostEqual(0.00311, pr.tau0().value, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
