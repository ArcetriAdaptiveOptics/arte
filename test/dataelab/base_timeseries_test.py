# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import unittest
import astropy.units as u
from astropy.io import fits
from arte.dataelab.base_timeseries import BaseTimeSeries

class BaseTimeSeriesTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range10x6.fits')
        self.testdata = fits.getdata(self.fitsfile)

    def test_data_without_unit_and_set_unit(self):
        series = BaseTimeSeries(self.testdata, astropy_unit = u.m)
        assert series.get_data().unit == u.m

    def test_data_without_unit_and_unset_unit(self):
        series = BaseTimeSeries(self.testdata, astropy_unit = None)
        assert not isinstance(series.get_data(), u.Quantity)

    def test_data_with_unit_and_same_unit(self):
        series = BaseTimeSeries(self.testdata*u.m, astropy_unit = u.m)
        assert series.get_data().unit == u.m

    def test_data_with_unit_and_different_unit(self):
        series = BaseTimeSeries(self.testdata*u.m, astropy_unit = u.cm)
        assert series.get_data().unit == u.cm

    def test_data_with_unit_and_incompatible_unit(self):
        with self.assertRaises(u.UnitsError):
            series = BaseTimeSeries(self.testdata*u.m, astropy_unit = u.kg)
            _ = series.get_data()

    def test_data_with_unit_and_unset_unit(self):
        series = BaseTimeSeries(self.testdata*u.m, astropy_unit = None)
        assert series.get_data().unit == u.m

    def test_get_display_expands_ensemble_dimension_to_2d(self):
        series = BaseTimeSeries(self.testdata*u.m, astropy_unit=None)
        assert series.get_display().shape == (10, 6, 1)
