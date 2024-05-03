# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import unittest
import numpy as np
import astropy.units as u
from arte.dataelab.base_pixels import BasePixels
from arte.dataelab.data_loader import FitsDataLoader

class BasePixelsTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range2x3.fits')

    def test_total_adu(self):
        px = BasePixels(FitsDataLoader(self.fitsfile))
        assert px.total_adu() == 7.5 * u.adu

    def test_display(self):
        data = np.arange(10*4*4).reshape(10,16)
        px = BasePixels(data)
        assert px.get_display().shape == (10, 4, 4)

    def test_nonsquare_display_raises(self):
        data = np.arange(10*4*6).reshape(10,24)
        px = BasePixels(data)
        with self.assertRaises(ValueError):
            _ = px.get_display()

