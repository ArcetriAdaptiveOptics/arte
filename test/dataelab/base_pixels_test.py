# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import unittest
import astropy.units as u
from arte.dataelab.base_pixels import BasePixels
from arte.dataelab.data_loader import FitsDataLoader

class BasePixelsTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range2x3.fits')

    def test_total_adu(self):
        px = BasePixels(1*u.s, FitsDataLoader(self.fitsfile))
        assert px.total_adu() == 7.5 * u.adu

