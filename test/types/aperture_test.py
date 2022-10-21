'''
Created on 2 feb 2022

@author: giuliacarla
'''

import unittest
import astropy.units as u
from arte.types.aperture import CircularOpticalAperture
from astropy.tests.helper import assert_quantity_allclose


class CircularOpticalApertureTest(unittest.TestCase):

    def testOutputAsInputPlusUnits(self):
        cen = [3, 1, 5]
        r = 4
        ape = CircularOpticalAperture(r, cen)
        want_cen = [3 * u.m, 1 * u.m, 5 * u.m]
        got_cen = ape.getCartesianCoords()
        want_r = 4 * u.m
        got_r = ape.getApertureRadius()
        assert_quantity_allclose(got_cen, want_cen)
        assert_quantity_allclose(got_r, want_r)

    def testInputsWithWrongUnits(self):
        cen = [300 * u.cm, 1 * u.m, 5000 * u.mm]
        r = 4e9 * u.nm
        ape = CircularOpticalAperture(r, cen)
        want_cen = [3 * u.m, 1 * u.m, 5 * u.m]
        got_cen = ape.getCartesianCoords()
        want_r = 4 * u.m
        got_r = ape.getApertureRadius()
        assert_quantity_allclose(got_cen, want_cen)
        assert_quantity_allclose(got_r, want_r)
