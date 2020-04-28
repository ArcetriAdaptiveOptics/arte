# -*- coding: utf-8 -*-

import numpy as np
import doctest
import unittest
import astropy.units as u

from arte.utils import astropy_utils

class AstropyUtilsTest(unittest.TestCase):

    def test_docstrings(self):
        doctest.testmod(astropy_utils, raise_on_error=True)
        
    def test_assert_array_almost_equal_w_units(self):
        
        a = np.arange(5)*u.mm
        b = np.arange(5)*u.mm
        c = np.arange(5)*u.mm +1*u.mm
        
        astropy_utils.assert_array_almost_equal_w_units(a, b)

        with self.assertRaises(AssertionError):
            astropy_utils.assert_array_almost_equal_w_units(a, c)

if __name__ == "__main__":
    unittest.main()
