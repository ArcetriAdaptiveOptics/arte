#!/usr/bin/env python
import doctest
import unittest
import numpy as np
from arte.utils.rebin import rebin


class RebinTest(unittest.TestCase):

    def setUp(self):
        self.ref = np.array([[0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1],
                             [2, 2, 2, 3, 3, 3],
                             [2, 2, 2, 3, 3, 3]])

    def test_docstring(self):
        from arte.utils import rebin
        doctest.testmod(rebin, raise_on_error=True)


    def test_newshape_types(self):

        a = np.arange(4).reshape((2,2))
        b = rebin(a, [4,6])
        np.testing.assert_array_equal(b, self.ref)

        c = rebin(a, map(int,'4 6'.split()))
        np.testing.assert_array_equal(c, self.ref)

        c = rebin(a, map(int,['4', 6]))
        np.testing.assert_array_equal(c, self.ref)

    def test_wrong_newshape(self):

        a = np.arange(4).reshape((2,2))

        with self.assertRaises(ValueError):
            _ = rebin(a, [4,6,8])

        with self.assertRaises(ValueError):
            _ = rebin(a, ['4.0',6])

    def test_up_sample(self):

        a = np.arange(4).reshape((2,2))
        b = rebin(a, (4,6), sample=True)
        np.testing.assert_array_equal(b, self.ref)

    def test_exceptions(self):

        a = np.arange(16).reshape((4,4))

        with self.assertRaises(ValueError):
            _ = rebin(a,(7,7))

        with self.assertRaises(ValueError):
            _ = rebin(a,(3,3))

        with self.assertRaises(NotImplementedError):
            _ = rebin(a,(3,7))
       

if __name__ == "__main__":
    unittest.main()
