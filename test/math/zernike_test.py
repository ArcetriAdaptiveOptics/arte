
import unittest
import numpy as np

from arte.math.zernike import Zernike


class ZernikeTest(unittest.TestCase):

    def test_resize(self):

        z1 = Zernike((2,10,10))
        z2 = Zernike((2,5,5))
        
        z2.resize((2, 10,10))

        np.testing.assert_array_equal(z1.get(), z2.get())

    def test_getitem(self):

        z1 = Zernike((2,10,10))

        np.testing.assert_array_equal(z1.get()[1], z1[1])


if __name__ == "__main__":
    unittest.main()
