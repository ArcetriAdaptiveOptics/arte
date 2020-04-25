# -*- coding: utf-8 -*-

import unittest
import astropy.units as u
from arte.types.aperture import CircularOpticalAperture


class CircularOpticalApertureTest(unittest.TestCase):


    def testNoUnits(self):
        
        a = CircularOpticalAperture(1, [1,2,3])
        
        assert a.getApertureRadius() == 1 * u.m
        assert a.getCartesianCoords() == [1*u.m, 2*u.m, 3*u.m]
        
    def testCorrectUnits(self):
        
        a = CircularOpticalAperture(1*u.m, [1*u.cm,2*u.nm,3*u.km])
        
        assert a.getApertureRadius() == 1 * u.m
        assert a.getCartesianCoords()[0] == 0.01 *u.m
        assert a.getCartesianCoords()[1] == 2e-9 *u.m
        assert a.getCartesianCoords()[2] == 3000 *u.m

    def testWrongUnits(self):
        
        with self.assertRaises(u.UnitsError):
            _ = CircularOpticalAperture(1*u.s, [1,2,3])

        with self.assertRaises(u.UnitsError):
            _ = CircularOpticalAperture(1, [1*u.ph,2,3])

                