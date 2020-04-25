# -*- coding: utf-8 -*-

import unittest
import astropy.units as u

from arte.utils.astropy_utils import get_the_unit_if_it_has_one, \
                                     match_and_remove_units

class AstropyUtilsTest(unittest.TestCase):


    def test_get_the_unit(self):
        
        a = 42
        b = 42*u.m
        
        assert get_the_unit_if_it_has_one(a) == 1
        assert get_the_unit_if_it_has_one(b) == u.m
        
    def test_match_and_remove_units(self):
        
        a = 42*u.m
        b = 42*u.cm
        c = 42*u.km
        
        a,b,c,d = match_and_remove_units(a,b,c)
        
        assert a==42
        assert b==0.42
        assert c==42000
        assert d==u.m

    def test_match_and_remove_units2(self):
        
        a = 42
        b = 42*u.m
        c = 42*u.cm
        
        a,b,c,d = match_and_remove_units(a,b,c)
        
        assert a==42
        assert b==42
        assert c==42
        assert d==1
   

if __name__ == "__main__":
    unittest.main()
