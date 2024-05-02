import os
import unittest
import astropy.units as u
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')   # Draw in background

from arte.dataelab.base_slopes import BaseSlopes

class BaseSlopesTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range10x6.fits')
        self.testdata = fits.getdata(self.fitsfile)
        mapper2d = lambda x: x.reshape((3,1))
        self.series = BaseSlopes(self.testdata*u.m, mapper2d=mapper2d, astropy_unit=None)

    def test_get_display_remaps_to_slopes_vector_to_2d(self):
        assert self.series.get_display().shape == (10, 3, 2)

    def test_vecshow(self):
        _ = self.series.vecshow()


if __name__ == "__main__":
    unittest.main()
