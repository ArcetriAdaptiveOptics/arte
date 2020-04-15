#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.radial_profile import computeRadialProfile

__version__ = "$Id:$"


class RadialProfileTest(unittest.TestCase):

    def setUp(self):
        self._mapFact= 42

    def _createDistanceMapSquareCenteredOn1Pixel(self):
        y, x= np.indices((10, 10))
        y= y - 5
        x= x - 5
        distanceMap= np.hypot(x, y)
        self.assertEqual(0, distanceMap[5, 5])
        return distanceMap * self._mapFact


    def testRadialProfileOnSquareEvenCenteredOn1PixelMap(self):
        image= self._createDistanceMapSquareCenteredOn1Pixel()
        profile, radialDistance= computeRadialProfile(image, 5, 5)
        self.assertAlmostEqual(0, profile[0])
        self.assertAlmostEqual(self._mapFact* (4 + 4* np.sqrt(2)) / 8,
                               profile[1])
        self.assertAlmostEqual(0, radialDistance[0])
        self.assertAlmostEqual((4 + 4* np.sqrt(2)) / 8,
                               radialDistance[1])
        self.assertTrue(np.allclose(radialDistance * self._mapFact,
                                    profile))


    def _createDistanceMapSquareCenteredOn4Pixels(self):
        y, x= np.indices((10, 10))
        y= y - 4.5
        x= x - 4.5
        distanceMap= np.hypot(x, y)
        self.assertEqual(np.hypot(0.5, 0.5),
                         distanceMap[5, 5])
        return distanceMap * self._mapFact


    def testRadialProfileOnSquareEvenCenteredOn4PixelMap(self):
        image= self._createDistanceMapSquareCenteredOn4Pixels()
        profile, radialDistance= computeRadialProfile(image, 4.5, 4.5)
        self.assertAlmostEqual(np.hypot(0.5, 0.5) * self._mapFact,
                               profile[0])
        self.assertAlmostEqual(np.hypot(0.5, 0.5), radialDistance[0])
        self.assertTrue(np.allclose(radialDistance * self._mapFact,
                                    profile))


    def _createDistanceMapSquareOn1PixelOffCenter(self):
        y, x= np.indices((10, 10))
        y= y - 4
        x= x - 5
        distanceMap= np.hypot(x, y)
        self.assertEqual(0, distanceMap[4, 5])
        return distanceMap * self._mapFact


    def testRadialProfileOnSquareEvenOn1PixelOffCenterMap(self):
        image= self._createDistanceMapSquareOn1PixelOffCenter()
        profile, radialDistance= computeRadialProfile(image, 4, 5)
        self.assertAlmostEqual(0, profile[0])
        self.assertAlmostEqual(self._mapFact* (4 + 4* np.sqrt(2)) / 8,
                               profile[1])
        self.assertAlmostEqual(0, radialDistance[0])
        self.assertAlmostEqual((4 + 4* np.sqrt(2)) / 8,
                               radialDistance[1])
        self.assertTrue(np.allclose(radialDistance * self._mapFact,
                                    profile))



if __name__ == "__main__":
    unittest.main()
