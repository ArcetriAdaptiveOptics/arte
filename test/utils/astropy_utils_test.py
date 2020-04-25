# -*- coding: utf-8 -*-

import doctest
import unittest
from arte.utils import astropy_utils

class AstropyUtilsTest(unittest.TestCase):

    def test_docstrings(self):
        doctest.testmod(astropy_utils, raise_on_error=True)
        

   

if __name__ == "__main__":
    unittest.main()
