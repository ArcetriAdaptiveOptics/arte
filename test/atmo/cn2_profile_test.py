#!/usr/bin/env python
import unittest
import numpy as np
from arte.atmo.cn2_profile import Cn2Profile, \
    MaorySteroScidarProfiles, EsoEltProfiles, MaoryStereoScidarProfiles2021
from arte.utils.constants import Constants
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose


class Cn2ProfileTest(unittest.TestCase):

    def testSingleLayer(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        self.assertAlmostEqual([100], pr.layers_distance().value)
        self.assertAlmostEqual(0.1, pr.r0().value)
        pr = Cn2Profile.from_fractional_j(
            0.12, [1.0], [30], [100], [10], [0.3])
        self.assertAlmostEqual(0.12, pr.r0().value)

    def testScalarAsArguments(self):
        pr = Cn2Profile.from_r0s(0.1, 30, 100, 10, 0.3)
        self.assertAlmostEqual(100, pr.layers_distance().value)
        self.assertAlmostEqual(0.1, pr.r0().value)
        pr = Cn2Profile.from_fractional_j(
            0.12, 1.0, 30, 100, 10, 0.3)
        self.assertAlmostEqual(0.12, pr.r0().value)

    def testSeeingAndR0ScaleWithWavelengths(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        pr.set_wavelength(500e-9)
        r0At500 = 0.1
        seeingAt500 = 0.98 * 500e-9 / r0At500 * Constants.RAD2ARCSEC
        self.assertAlmostEqual(r0At500, pr.r0().value)
        self.assertAlmostEqual(seeingAt500, pr.seeing().value)

        pr.set_wavelength(2200e-9)
        self.assertAlmostEqual(r0At500 * (2200 / 500) **
                               (6. / 5), pr.r0().value)
        self.assertAlmostEqual(seeingAt500 * (2200 / 500) ** (-1. / 5),
                               pr.seeing().value)

    def testSetAndGetAirmassAndZenithAngle(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        pr.set_zenith_angle(0)
        self.assertAlmostEqual(0, pr.zenith_angle().value)
        self.assertAlmostEqual(1, pr.airmass().value)
        pr.set_zenith_angle(60)
        self.assertAlmostEqual(60, pr.zenith_angle().value)
        self.assertAlmostEqual(2, pr.airmass().value)

    def testSetAndGetWindSpeedAndDirection(self):
        pr = Cn2Profile.from_r0s([0.1], [30], [100], [10], [0.3])
        pr.set_wind_speed(50)
        self.assertAlmostEqual(50, pr.wind_speed().value)
        pr.set_wind_direction(90)
        self.assertAlmostEqual(90, pr.wind_direction().value)

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
        self.assertAlmostEqual(np.inf, pr.theta0())

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
        pr = MaorySteroScidarProfiles.P10()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.48, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2.7, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(5.18, pr.wind_speed()[0].value, delta=0.1)
        self.assertAlmostEqual(0, pr.wind_direction()[0].value)

        pr = MaorySteroScidarProfiles.P25()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.55, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2.2, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(5.3, pr.wind_speed()[0].value, delta=0.1)

        pr = MaorySteroScidarProfiles.P50()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.66, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.74, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(5.7, pr.wind_speed()[0].value, delta=0.1)

        pr = MaorySteroScidarProfiles.P75()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.78, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.32, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(6.2, pr.wind_speed()[0].value, delta=0.1)

        pr = MaorySteroScidarProfiles.P90()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.92, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.04, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(6.9, pr.wind_speed()[0].value, delta=0.1)

    def testMaory2021PercentileProfiles(self):
        pr = MaoryStereoScidarProfiles2021.P10()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.509, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2.82, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(8.09, pr.mean_wind_speed().value, delta=0.1)
        self.assertAlmostEqual(7.72, pr.tau0().value * 1e3, delta=0.1)

        pr = MaoryStereoScidarProfiles2021.P25()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.603, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2.43, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(8.94, pr.mean_wind_speed().value, delta=0.1)
        self.assertAlmostEqual(5.84, pr.tau0().value * 1e3, delta=0.1)

        pr = MaoryStereoScidarProfiles2021.P50()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.725, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(2., pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(10.3, pr.mean_wind_speed().value, delta=0.1)
        self.assertAlmostEqual(4.21, pr.tau0().value * 1e3, delta=0.1)

        pr = MaoryStereoScidarProfiles2021.P75()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.889, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(1.62, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(12.2, pr.mean_wind_speed().value, delta=0.1)
        self.assertAlmostEqual(2.91, pr.tau0().value * 1e3, delta=0.1)

        pr = MaoryStereoScidarProfiles2021.P90()
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(1.05, pr.seeing().value, delta=0.02)
        self.assertAlmostEqual(1.35, pr.theta0().value, delta=0.1)
        self.assertAlmostEqual(14.4, pr.mean_wind_speed().value, delta=0.1)
        self.assertAlmostEqual(2.05, pr.tau0().value * 1e3, delta=0.1)

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
        self.assertAlmostEqual(35, pr.number_of_layers())
        self.assertAlmostEqual(0.644, pr.seeing().value, delta=0.01)
        self.assertAlmostEqual(5.786, pr.wind_speed()[0].value, delta=0.01)
        self.assertAlmostEqual(0.00535, pr.tau0().value, delta=0.0001)
        self.assertAlmostEqual(9.2116, pr.mean_wind_speed().value, delta=0.01)
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

    def testEsoProfilesCanModifyL0(self):
        pr = EsoEltProfiles.Median(L0=np.inf)
        self.assertAlmostEqual(np.inf, pr.outer_scale().value[0], delta=0.001)

# TODO: [0.3 * u.deg] as wind direction input doesn't work because
# it is seen as a list, so it cannot be converted in rad using np.deg2rad
#     def testScalarQuantityAsArgument(self):
#         pr = Cn2Profile.from_r0s([100 * u.mm],
#                                  [30 * u.m],
#                                  [0.1 * u.km],
#                                  [10 * u.m / u.s],
#                                  [0.3 * u.deg])
#         assert_quantity_allclose(100 * u.m, pr.layers_distance())
#         assert_quantity_allclose(0.1 * u.m, pr.r0())

    def testQuantityAsArgument(self):
        pr = Cn2Profile.from_r0s([100 * u.mm, 0.2 * u.m],
                                 np.array([30, 50]) * u.m,
                                 [0.1 * u.km, 10000 * u.m],
                                 [10, 10] * u.m / u.s,
                                 [0.3, 45] * u.deg)
        assert_quantity_allclose([100, 10000] * u.m, pr.layers_distance())
        assert_quantity_allclose([0.1, 0.2] * u.m, pr.r0s())

    def testSameOutputWithOrWithoutQuantityAsInput(self):
        prWithQuantity = Cn2Profile.from_r0s([0.1] * u.m, [25] * u.m,
                                             [10000] * u.m, [10] * u.m / u.s,
                                             [0.3] * u.deg)
        prWithoutQuantity = Cn2Profile.from_r0s([0.1], [25], [10000], [10],
                                                [0.3])
        assert_quantity_allclose(prWithQuantity.r0(),
                                 prWithoutQuantity.r0())
        assert_quantity_allclose(prWithQuantity.outer_scale(),
                                 prWithoutQuantity.outer_scale())
        assert_quantity_allclose(prWithQuantity.layers_distance(),
                                 prWithoutQuantity.layers_distance())
        assert_quantity_allclose(prWithQuantity.wind_speed(),
                                 prWithoutQuantity.wind_speed())
        assert_quantity_allclose(prWithQuantity.wind_direction(),
                                 prWithoutQuantity.wind_direction())
        assert_quantity_allclose(prWithQuantity.theta0(),
                                 prWithoutQuantity.theta0())
        assert_quantity_allclose(prWithQuantity.tau0(),
                                 prWithoutQuantity.tau0())
        assert_quantity_allclose(prWithQuantity.seeing(),
                                 prWithoutQuantity.seeing())
        assert_quantity_allclose(prWithQuantity.zenith_angle(),
                                 prWithoutQuantity.zenith_angle())


if __name__ == "__main__":
    unittest.main()
