'''
Created on 13 dic 2019

@author: giuliacarla
'''

import unittest
import numpy as np
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

    def testInputsWithWrongUnits(self):
        rho, theta = (5000 * u.marcsec, np.pi / 4 * u.rad)
        z = 100e3 * u.cm
        source = GuideSource((rho, theta), z)
        want_polar = [5 * u.arcsec, 45 * u.deg, 1000 * u.m]
        got_polar = source.getSourcePolarCoords()
        want_cartes = [5 * np.sqrt(2) / 2 * u.arcsec,
                       5 * np.sqrt(2) / 2 * u.arcsec,
                       1000 * u.m]
        got_cartes = source.getSourceCartesianCoords()
        assert_quantity_allclose(got_polar[0], want_polar[0])
        assert_quantity_allclose(got_polar[1], want_polar[1])
        assert_quantity_allclose(got_polar[2], want_polar[2])
        assert_quantity_allclose(got_cartes[0], want_cartes[0])
        assert_quantity_allclose(got_cartes[1], want_cartes[1])
        assert_quantity_allclose(got_cartes[2], want_cartes[2])

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
