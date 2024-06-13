# -*- coding: utf-8 -*-

import doctest
import unittest
import numpy as np
import astropy.units as u
from arte.utils.unit_checker import unit_check, make_sure_its_a

class UnitCheckTest(unittest.TestCase):

    @unit_check
    def a(self, a=1,
                b=2* u.s,
                c=3* u.m):
        '''function a'''
        assert not isinstance(a, u.Quantity)
        assert b.unit == u.s
        assert c.unit == u.m

    @unit_check
    def b(self, a=1):
        assert isinstance(a, u.Quantity)

    def test_nothing(self):
        self.a()
        
    def test_wraps(self):
        assert self.a.__doc__ == 'function a'

    def test_args_without_units_are_not_touched(self):
        self.a(42)
        self.b(42*u.m)

    def test_defaults_set_their_units(self):
        self.a(1,2,3)

    def test_units_are_converted(self):
        self.a(1, 42*u.year, c=3*u.nm)

    def test_wrong_units(self):
        
        with self.assertRaises(u.UnitsError):
            self.a(c=2*u.s)

        with self.assertRaises(u.UnitsError):
            self.a(b=2*u.m)

    def test_docstrings(self):
        from arte.utils import unit_checker
        doctest.testmod(unit_checker, raise_on_error=True)

    def test_make_sure_its_a_with_masked_array_returns_masked_array(self):
            mask = np.array([[True, False], [False, False]])
            data = np.arange(4).reshape((2, 2))
            v = np.ma.array(data, mask=mask)
            c = make_sure_its_a(u.m, v)
            self.assertTrue(isinstance(c, np.ma.MaskedArray))
            self.assertFalse(np.isnan(c).data.value.any())

    def test_make_sure_its_a_with_masked_array_and_unit_returns_masked_array(self):
            mask = np.array([[True, False], [False, False]])
            data = np.arange(4).reshape((2, 2)) * u.cm
            v = np.ma.array(data, mask=mask)
            c = make_sure_its_a(u.m, v)
            self.assertTrue(isinstance(c, np.ma.MaskedArray))
            self.assertTrue(isinstance(c.data, u.Quantity))
            assert c.data.unit == u.m
            np.testing.assert_array_almost_equal(c.data.value,data.value * 0.01)
