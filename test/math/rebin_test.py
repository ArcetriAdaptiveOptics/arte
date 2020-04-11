
import unittest
import numpy as np

from arte.math.rebin import rebin2d

class RebinTest(unittest.TestCase):

    def test_upsample(self):
        '''
        '''
        a = np.arange(4).reshape((2,2))
        b = rebin2d(a, (4,4), sample=True)

        c = np.array([[ 0, 0, 1, 1 ],
                      [ 0, 0, 1, 1 ],
                      [ 2, 2, 3, 3 ],
                      [ 2, 2, 3, 3 ]])

        assert np.allclose(b,c)

    def test_downsample(self):

        a = np.arange(16).reshape((4,4))
        b = rebin2d(a,(2,2))
        
        c = np.array([[ 2.5,  4.5, ],
                      [ 10.5, 12.5]])
    
        assert np.allclose(b,c)

if __name__ == "__main__":
    unittest.main()

