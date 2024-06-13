# -*- coding: utf-8 -*-

#!/usr/bin/env python
import os
import unittest
import numpy as np
import astropy.units as u
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')   # Draw in background

from arte.dataelab.data_loader import FitsDataLoader, NumpyDataLoader
from arte.dataelab.base_timeseries import BaseTimeSeries

class BaseTimeSeriesTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range10x6.fits')
        self.testdata = fits.getdata(self.fitsfile)
        self.rangefile = os.path.join(mydir, 'testdata', 'range1to5.npy')

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

    def test_get_display_removes_units(self):
        series = BaseTimeSeries(self.testdata*u.s, astropy_unit=None)
        assert not isinstance(series.get_display(), u.Quantity)

    def test_get_display_expands_ensemble_dimension_to_3d(self):
        series = BaseTimeSeries(self.testdata, astropy_unit=None)
        assert series.get_display().shape == (10, 6, 1)

    def test_plot_hist(self):
        series = BaseTimeSeries(self.testdata, astropy_unit=None)
        _ = series.plot_hist()

    def test_plot_spectra(self):
        series = BaseTimeSeries(self.testdata, astropy_unit=None)
        _ = series.plot_spectra()

    def test_plot_cumulative_spectra(self):
        series = BaseTimeSeries(self.testdata, astropy_unit=None)
        _ = series.plot_cumulative_spectra()

    def test_loader_filename(self):
        series = BaseTimeSeries(FitsDataLoader(self.fitsfile))
        assert series.filename().endswith('range10x6.fits')

    def test_filename_without_loader(self):
        series = BaseTimeSeries(self.fitsfile)
        assert series.filename().endswith('range10x6.fits')

    def test_wrong_filename1(self):
        with self.assertRaises(ValueError):
            _ = BaseTimeSeries(None)

    def test_wrong_filename2(self):
        with self.assertRaises(ValueError):
            _ = BaseTimeSeries(dict())

    def test_timevector_filename(self):
        series = BaseTimeSeries(self.testdata, time_vector=NumpyDataLoader(self.rangefile))
        np.testing.assert_array_equal(series.get_time_vector(), range(1, 5))

    def test_timevector_filename_without_loader(self):
        series = BaseTimeSeries(self.testdata, time_vector=self.rangefile)
        np.testing.assert_array_equal(series.get_time_vector(), range(1, 5))

    def test_timevector_wrong_filename1(self):
        with self.assertRaises(ValueError):
            _ = BaseTimeSeries(self.testdata, time_vector=set())


class TestBaseTimeSeriesMath(unittest.TestCase):
    '''Written by ChaptGPT, correct on the first try'''

    def test_addition_numpy_arrays(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]))
        result = a + b
        np.testing.assert_array_equal(result.get_data(), np.array([[6, 8], [10, 12]]))

    def test_subtraction_numpy_arrays(self):
        a = BaseTimeSeries(np.array([[9, 8], [7, 6]]))
        b = BaseTimeSeries(np.array([[5, 4], [3, 2]]))
        result = a - b
        np.testing.assert_array_equal(result.get_data(), np.array([[4, 4], [4, 4]]))

    def test_multiplication_numpy_arrays(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]))
        result = a * b
        np.testing.assert_array_equal(result.get_data(), np.array([[5, 12], [21, 32]]))

    def test_division_numpy_arrays(self):
        a = BaseTimeSeries(np.array([[8, 12], [18, 24]]))
        b = BaseTimeSeries(np.array([[2, 3], [6, 8]]))
        result = a / b
        np.testing.assert_array_equal(result.get_data(), np.array([[4, 4], [3, 3]]))

    def test_addition_astropy_quantities(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]) * u.m)
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]) * u.m)
        result = a + b
        np.testing.assert_array_equal(result.get_data().value, np.array([[6, 8], [10, 12]]))
        self.assertEqual(result.get_data().unit, u.m)

    def test_addition_astropy_quantities_with_numpy_array(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]) * u.m)
        with self.assertRaises(u.UnitConversionError):
            _ = a + b

    def test_addition_numpy_array_with_astropy_quantities(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]) * u.m)
        with self.assertRaises(u.UnitConversionError):
            _ = a + b

    def test_subtraction_astropy_quantities(self):
        a = BaseTimeSeries(np.array([[9, 8], [7, 6]]) * u.m)
        b = BaseTimeSeries(np.array([[5, 4], [3, 2]]) * u.m)
        result = a - b
        np.testing.assert_array_equal(result.get_data().value, np.array([[4, 4], [4, 4]]))
        self.assertEqual(result.get_data().unit, u.m)

    def test_multiplication_astropy_quantities(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]) * u.m)
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]))
        result = a * b
        np.testing.assert_array_equal(result.get_data().value, np.array([[5, 12], [21, 32]]))
        self.assertEqual(result.get_data().unit, u.m)

    def test_division_astropy_quantities(self):
        a = BaseTimeSeries(np.array([[8, 12], [18, 24]]) * u.m)
        b = BaseTimeSeries(np.array([[2, 3], [6, 8]]))
        result = a / b
        np.testing.assert_array_equal(result.get_data().value, np.array([[4, 4], [3, 3]]))
        self.assertEqual(result.get_data().unit, u.m)

    def test_addition_mismatched_units(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]) * u.m)
        b = BaseTimeSeries(np.array([[5, 6], [7, 8]]) * u.s)
        with self.assertRaises(u.UnitConversionError):
            _ = a + b

    def test_subtraction_mismatched_units(self):
        a = BaseTimeSeries(np.array([[9, 8], [7, 6]]) * u.m)
        b = BaseTimeSeries(np.array([[5, 4], [3, 2]]) * u.s)
        with self.assertRaises(u.UnitConversionError):
            _ = a - b

    def test_invalid_addition(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            _ = a + "invalid"

    def test_invalid_subtraction(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            _ = a - "invalid"

    def test_invalid_multiplication(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            _ = a * "invalid"

    def test_invalid_division(self):
        a = BaseTimeSeries(np.array([[1, 2], [3, 4]]))
        with self.assertRaises(TypeError):
            _ = a / "invalid"