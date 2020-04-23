# -*- coding: utf-8 -*-

import unittest
import astropy.units as u
from arte.utils.unit_checker import unit_check

class UnitCheckTest(unittest.TestCase):

    @unit_check
    def a(self, a=1,
                b=2* u.s,
                c=3* u.m):

        assert not isinstance(a, u.Quantity)
        assert b.unit == u.s
        assert c.unit == u.m

    @unit_check
    def b(self, a=1):
        assert isinstance(a, u.Quantity)

    def test_nothing(self):
        self.a()

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
