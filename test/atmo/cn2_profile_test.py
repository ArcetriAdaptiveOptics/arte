#!/usr/bin/env python
import unittest
import numpy as np
from apposto.atmo.cn2_profile import Cn2Profile, MaoryProfiles
from apposto.utils.constants import Constants


class Cn2ProfileTest(unittest.TestCase):


    def testSingleLayer(self):
        pr= Cn2Profile.fromR0s([0.1], [30], [100], [10], [0.3])
        self.assertEqual([100], pr.layersDistance())
        self.assertEqual(0.1, pr.r0())
        pr= Cn2Profile.fromFractionalJ(
            0.12, [1.0], [30], [100], [10], [0.3])
        self.assertEqual(0.12, pr.r0())


    def testScalarAsArguments(self):
        pr= Cn2Profile.fromR0s(0.1, 30, 100, 10, 0.3)
        self.assertEqual(100, pr.layersDistance())
        self.assertEqual(0.1, pr.r0())
        pr= Cn2Profile.fromFractionalJ(
            0.12, 1.0, 30, 100, 10, 0.3)
        self.assertEqual(0.12, pr.r0())


    def testSeeingAndR0ScaleWithWavelengths(self):
        pr= Cn2Profile.fromR0s([0.1], [30], [100], [10], [0.3])
        pr.setWavelength(500e-9)
        r0At500=0.1
        seeingAt500=0.98 * 500e-9 / r0At500 * Constants.RAD2ARCSEC
        self.assertEqual(r0At500, pr.r0())
        self.assertEqual(seeingAt500, pr.seeing())

        pr.setWavelength(2200e-9)
        self.assertEqual(r0At500 * (2200/500)**(6./5), pr.r0())
        self.assertEqual(seeingAt500*(2200/500)**(-1./5),
                         pr.seeing())


    def testSetAndGetAirmassAndZenithAngle(self):
        pr= Cn2Profile.fromR0s([0.1], [30], [100], [10], [0.3])
        pr.setZenithAngle(0)
        self.assertAlmostEqual(0, pr.zenithAngle())
        self.assertAlmostEqual(1, pr.airmass())
        pr.setZenithAngle(60)
        self.assertAlmostEqual(60, pr.zenithAngle())
        self.assertAlmostEqual(2, pr.airmass())


    def testBuildFromFractionalJ(self):
        pr= Cn2Profile.fromFractionalJ(0.1, 1.0, 30, 100, 10, 0.3)
        self.assertAlmostEqual(0.1, pr.r0())
        pr= Cn2Profile.fromFractionalJ(0.13, [0.1, 0.9], 30, 100, 10, 0.3)
        self.assertAlmostEqual(0.13, pr.r0())

        dontSumToOne= np.array([0.1, 42])
        self.assertRaises(Exception, Cn2Profile.fromFractionalJ,
                          42, dontSumToOne, 30, 100, 10, 0)


    def testR0ScaleWithAirmass(self):
        pr= Cn2Profile.fromFractionalJ(0.1, 1.0, 30, 100, 10, 0.3)
        pr.setZenithAngle(0)
        self.assertAlmostEqual(0.1, pr.r0())
        pr.setZenithAngle(60)
        self.assertAlmostEqual(0.1*2**(-3./5), pr.r0())
        pr.setZenithAngle(45)
        self.assertAlmostEqual(0.1*np.sqrt(2)**(-3./5), pr.r0())


    def testTheta0AtGroundIsInfinite(self):
        pr= Cn2Profile.fromFractionalJ(0.1, 1.0, 30, 0, 10, 0.3)
        pr.setZenithAngle(42)
        self.assertEqual(np.inf, pr.theta0())


    def testTheta0OfSingleLayer(self):
        zenAngle= 23.5
        airmass= Cn2Profile.zenithAngle2Airmass(zenAngle)
        r0= 0.13
        h= 1000.
        pr= Cn2Profile.fromFractionalJ(r0, 1.0, 30, h, 10, 0.3)
        pr.setZenithAngle(zenAngle)
        pr.setWavelength(1.0e-6)
        wanted= (2.914/0.422727)**(-3./5) * pr.r0() / (
            h*airmass) * Constants.RAD2ARCSEC
        self.assertAlmostEqual(wanted, pr.theta0())


    def testAirmass(self):
        self.assertAlmostEqual(1, Cn2Profile.zenithAngle2Airmass(0))
        self.assertAlmostEqual(2, Cn2Profile.zenithAngle2Airmass(60))
        self.assertAlmostEqual(np.sqrt(2),
                               Cn2Profile.zenithAngle2Airmass(45))


    def testMaoryPercentileProfiles(self):
        pr= MaoryProfiles.P10()
        self.assertEqual(35, pr.numberOfLayers())
        self.assertAlmostEqual(0.48, pr.seeing(), delta=0.01)
        self.assertAlmostEqual(2.7, pr.theta0(), delta=0.1)
        self.assertAlmostEqual(5.18, pr.windSpeed()[0], delta=0.1)
        self.assertAlmostEqual(0, pr.windDirection()[0])

        pr= MaoryProfiles.P25()
        self.assertEqual(35, pr.numberOfLayers())
        self.assertAlmostEqual(0.55, pr.seeing(), delta=0.01)
        self.assertAlmostEqual(2.2, pr.theta0(), delta=0.1)
        self.assertAlmostEqual(5.3, pr.windSpeed()[0], delta=0.1)

        pr= MaoryProfiles.P50()
        self.assertEqual(35, pr.numberOfLayers())
        self.assertAlmostEqual(0.66, pr.seeing(), delta=0.01)
        self.assertAlmostEqual(1.74, pr.theta0(), delta=0.1)
        self.assertAlmostEqual(5.7, pr.windSpeed()[0], delta=0.1)

        pr= MaoryProfiles.P75()
        self.assertEqual(35, pr.numberOfLayers())
        self.assertAlmostEqual(0.78, pr.seeing(), delta=0.01)
        self.assertAlmostEqual(1.32, pr.theta0(), delta=0.1)
        self.assertAlmostEqual(6.2, pr.windSpeed()[0], delta=0.1)

        pr= MaoryProfiles.P90()
        self.assertEqual(35, pr.numberOfLayers())
        self.assertAlmostEqual(0.92, pr.seeing(), delta=0.01)
        self.assertAlmostEqual(1.04, pr.theta0(), delta=0.1)
        self.assertAlmostEqual(6.9, pr.windSpeed()[0], delta=0.1)


    def testTau0OfSingleLayer(self):
        zenAngle= 10.
        r0= 0.13
        h= 1234.
        wspd= 10.
        pr= Cn2Profile.fromFractionalJ(r0, 1.0, 30, h, wspd, 0.3)
        pr.setZenithAngle(zenAngle)
        pr.setWavelength(1.0e-6)
        wanted= (2.914/0.422727)**(-3./5) * pr.r0() / (
            wspd)
        self.assertAlmostEqual(wanted, pr.tau0())


if __name__ == "__main__":
    unittest.main()