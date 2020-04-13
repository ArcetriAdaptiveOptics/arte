
import unittest
import numpy as np

from arte.math.make_xy import make_xy

class MakeXYTest(unittest.TestCase):

    def test_xy(self):
        '''Test that x and y are transposed wrt. each other'''

        x,y = make_xy(2,2)

        assert np.all(x.T == y)

if __name__ == "__main__":
    unittest.main()

