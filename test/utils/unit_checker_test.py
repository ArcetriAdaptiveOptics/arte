# -*- coding: utf-8 -*-

import doctest
import unittest
import numpy as np
import astropy.units as u
from arte.utils.unit_checker import unit_check
from arte.utils.unit_checker import assert_array_almost_equal_w_units

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
        
    def test_assert_array_almost_equal_w_units(self):
        
        a = np.arange(5)*u.mm
        b = np.arange(5)*u.mm
        c = np.arange(5)*u.mm +1*u.mm
        
        assert_array_almost_equal_w_units(a, b)

        with self.assertRaises(AssertionError):
            assert_array_almost_equal_w_units(a, c)