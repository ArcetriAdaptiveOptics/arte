#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.rebin import rebin


class RebinTest(unittest.TestCase):

    def setUp(self):
        self.ref = np.array([[0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1],
                             [2, 2, 2, 3, 3, 3],
                             [2, 2, 2, 3, 3, 3]])
        self.downsample_ref1 = np.array([[0., 0.5, 1.],
                                         [2., 2.5, 3.]])
        self.downsample_ref2 = np.array([[0, 0, 1],
                                         [2, 2, 3]])

    def test_up(self):

        a = np.arange(4).reshape((2,2))
        b = rebin(a, (4,6))
        np.testing.assert_array_equal(b, self.ref)

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

    def test_down(self):

        c = rebin(self.ref, (2, 3))
        np.testing.assert_array_equal(c, self.downsample_ref1)

    def test_down_sample(self):

        d = rebin(self.ref, (2, 3), sample=True)
        np.testing.assert_array_equal(d, self.downsample_ref2)

    def test_exceptions(self):

        a = np.arange(16).reshape((4,4))

        with self.assertRaises(ValueError):
            _ = rebin(a,(7,7))

        with self.assertRaises(ValueError):
            _ = rebin(a,(3,3))

        with self.assertRaises(ValueError):
            _ = rebin(a,(3,7))
       

if __name__ == "__main__":
    unittest.main()
