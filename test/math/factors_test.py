# -*- coding: utf-8 -*-
import unittest

from arte.math.factors import factors, gcd, lcm

class FactorsTest(unittest.TestCase):

    def test_factors(self):
        
        assert factors(8) == [2,4]
        assert factors(17) == []
        
    def test_gcd(self):

        assert gcd(15,20) == 5
        assert gcd(4,17) == 1

    def test_lcm(self):
        
        assert lcm(5,6) == 30
        assert lcm(10,20) == 20

if __name__ == "__main__":
    unittest.main()

