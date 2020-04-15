'''
Created on 13 dic 2019

@author: giuliacarla
'''

import unittest
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from arte.types.guide_source import GuideSource


class GuideSourceTest(unittest.TestCase):

    def testOutputAsInputPlusUnits(self):
        rho, theta = (50, 90)
        z = 100e3
        source = GuideSource((rho, theta), z)

        want = [50 * u.arcsec, 90 * u.deg, 100e3 * u.m]
        got = source.getSourcePolarCoords()
        assert_quantity_allclose(got[0], want[0])
        assert_quantity_allclose(got[1], want[1])
        assert_quantity_allclose(got[2], want[2])

    def testCartesianCoords(self):
        rho, theta = (50, 90)
        z = 100e3
        source = GuideSource((rho, theta), z)

        want = [0 * u.arcsec, 50 * u.arcsec, 100e3 * u.m]
        got = source.getSourceCartesianCoords()
        assert_quantity_allclose(got[0], want[0], atol=1e-14 * u.arcsec)
        assert_quantity_allclose(got[1], want[1])
        assert_quantity_allclose(got[2], want[2])

    def testQuantiyAsArgumentInPolarCoords(self):
        rho, theta = (50 * u.arcsec, 90 * u.deg)
        z = 100e3 * u.m
        source = GuideSource((rho, theta), z)

        want = [50 * u.arcsec, 90 * u.deg, 100e3 * u.m]
        got = source.getSourcePolarCoords()
        assert_quantity_allclose(got[0], want[0])
        assert_quantity_allclose(got[1], want[1])
        assert_quantity_allclose(got[2], want[2])

    def testQuantityAsArgumentInCartesianCoords(self):
        rho, theta = (50 * u.arcsec, 90 * u.deg)
        z = 100e3 * u.m
        source = GuideSource((rho, theta), z)

        want = [0 * u.arcsec, 50 * u.arcsec, 100e3 * u.m]
        got = source.getSourceCartesianCoords()
        assert_quantity_allclose(got[0], want[0], atol=1e-14 * u.arcsec)
        assert_quantity_allclose(got[1], want[1])
        assert_quantity_allclose(got[2], want[2])
