
import unittest
import numpy as np

from arte.math.toccd import toccd


class ToCcdTest(unittest.TestCase):

    def test_toccd(self):
        a = np.arange(16).reshape((4,4))
        b = toccd(a, (4,2))

        c = np.array([[1, 5],
                      [9, 13],
                      [17, 21],
                      [25, 29]])

        assert np.allclose(b,c)

    def test_total(self):

        a = np.arange(16).reshape((4,4))
        b = toccd(a, (4,2), set_total=12)

        c = np.array([[0.1, 0.5],
                      [0.9, 1.3],
                      [1.7, 2.1],
                      [2.5, 2.9]])

        assert np.allclose(b,c)

if __name__ == "__main__":
    unittest.main()

