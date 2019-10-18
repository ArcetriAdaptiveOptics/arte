#!/usr/bin/env python
import unittest
from apposto.utils.math import round_up_to_even


class RoundToEvenTest(unittest.TestCase):


    def testRoundUpToEven(self):
        self.assertEqual(2, round_up_to_even(2))
        self.assertEqual(2, round_up_to_even(0.1))
        self.assertEqual(4, round_up_to_even(3.1))
        self.assertEqual(0, round_up_to_even(0))
        self.assertEqual(0, round_up_to_even(-1.9))
        self.assertEqual(-2, round_up_to_even(-3))


if __name__ == "__main__":
    unittest.main()
