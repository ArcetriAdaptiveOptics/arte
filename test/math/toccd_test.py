
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

    def test_same(self):
        
        a = np.arange(16).reshape((4, 4))
        b = toccd(a, (4, 4))
        assert a is b

    def test_wrong_input_dims(self):
        
        a = np.arange(8).reshape((2, 2, 2))
        with self.assertRaises(ValueError):
            _ = toccd(a, (4, 4))

    def test_wrong_output_dims(self):
        
        a = np.arange(4).reshape((2, 2))
        with self.assertRaises(ValueError):
            _ = toccd(a, (2, 2, 2))

if __name__ == "__main__":
    unittest.main()

