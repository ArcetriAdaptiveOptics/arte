# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import unittest
import astropy.units as u
from astropy.io import fits
from arte.dataelab.base_data import BaseData

class BaseTimeSeriesTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range2x3.fits')
        self.testdata = fits.getdata(self.fitsfile)

    def test_data_without_unit_and_set_unit(self):
        dataobj = BaseData(self.testdata, astropy_unit = u.m)
        assert dataobj.get_data().unit == u.m

    def test_data_without_unit_and_unset_unit(self):
        dataobj = BaseData(self.testdata, astropy_unit = None)
        assert not isinstance(dataobj.get_data(), u.Quantity)

    def test_data_with_unit_and_same_unit(self):
        dataobj = BaseData(self.testdata*u.m, astropy_unit = u.g)
        assert dataobj.get_data().unit == u.g

    def test_data_with_unit_and_different_unit(self):
        dataobj = BaseData(self.testdata*u.m, astropy_unit = u.m)
        assert dataobj.get_data().unit == u.m

    def test_data_with_unit_and_unset_unit(self):
        dataobj = BaseData(self.testdata*u.m, astropy_unit = None)
        assert dataobj.get_data().unit == u.m
