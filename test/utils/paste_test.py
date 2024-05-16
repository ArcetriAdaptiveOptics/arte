import unittest
import numpy as np

from arte.utils.paste import paste


class PasteTest(unittest.TestCase):

    def test_one_dimensional(self):
        b = np.zeros([10])
        a = np.arange(1, 5)
        paste(b, a, (8,))
        want = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.])
        np.testing.assert_equal(b, want)

    def test_bidimensional(self):
        b = np.zeros([7, 7])
        a = np.arange(1, 33).reshape(4, 8)
        paste(b, a, (-1, -3))
        want = np.array([[12.,  13.,  14.,  15.,  16.,  0.,   0.],
                         [20.,  21.,  22.,  23.,  24.,  0.,   0.],
                         [28.,  29.,  30.,  31.,  32.,  0.,   0.],
                         [0.,   0.,   0.,   0.,   0.,   0.,   0.],
                         [0.,   0.,   0.,   0.,   0.,   0.,   0.],
                         [0.,   0.,   0.,   0.,   0.,   0.,   0.],
                         [0.,   0.,   0.,   0.,   0.,   0.,   0.]])
        np.testing.assert_equal(b, want)

    def test_completely_out(self):
        b = np.zeros([7, 7])
        a = np.ones((12, 23))
        paste(b, a, (10, 15))
        np.testing.assert_equal(b, np.zeros([7, 7]))


if __name__ == "__main__":
    unittest.main()
