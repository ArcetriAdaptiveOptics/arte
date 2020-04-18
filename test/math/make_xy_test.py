
import unittest
import numpy as np

from arte.math.make_xy import make_xy

class MakeXYTest(unittest.TestCase):

    def test_xy(self):
        '''Test that x and y are transposed wrt. each other'''

        x,y = make_xy(2,2)

        assert np.all(x.T == y)

    def test_sampling_fails(self):
        
        with self.assertRaises(ValueError):
            make_xy(1,2)

    def test_vector_sampling(self):
        
        ref = [-0.75, -0.25, 0.25, 0.75]
        arr = make_xy(4, 1, vector=True)
        np.testing.assert_array_equal( arr, ref)

    def test_zero_sampled(self):
        
        ref = [-1, -0.5, 0, 0.5]
        arr = make_xy(4, 1, vector=True, zero_sampled=True)
        np.testing.assert_array_equal( arr, ref)

    def test_odd_sampling(self):
        
        ref = [-0.8, -0.4, 0, 0.4, 0.8]
        arr1 = make_xy(5, 1, vector=True)
        arr2 = make_xy(5, 1, vector=True, zero_sampled=True)
        
        np.testing.assert_array_equal( arr1, ref)
        np.testing.assert_array_equal( arr2, ref)
        
if __name__ == "__main__":
    unittest.main()

