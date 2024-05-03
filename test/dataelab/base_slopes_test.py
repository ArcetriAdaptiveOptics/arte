import os
import unittest
import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')   # Draw in background

from arte.dataelab.base_slopes import BaseSlopes


def slope_for_3subaps_in_2d(x):
    '''2d slope mapper for a sensor with 3 subaps'''
    nsubaps = 3
    if len(x) == nsubaps*2:
        return x.reshape(nsubaps,2)
    elif len(x) == nsubaps:
        return np.atleast_2d(x)
    else:
        raise ValueError(f'Number of slopes must be either {nsubaps} or {nsubaps*2}')


class BaseSlopesTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range10x6.fits')
        self.testdata = fits.getdata(self.fitsfile)
        self.series = BaseSlopes(self.testdata, mapper2d=slope_for_3subaps_in_2d, astropy_unit=None)

    def test_get_display_remaps_to_slopes_vector_to_2d(self):
        assert self.series.get_display().shape == (10, 3, 2)

    def test_get_display_single_slope(self):
        assert self.series.get_display('x').shape == (10, 1, 3)
        assert self.series.get_display('y').shape == (10, 1, 3)

    def test_vecshow(self):
        _ = self.series.vecshow()


if __name__ == "__main__":
    unittest.main()
