import os
import unittest
import astropy.units as u
from astropy.io import fits
from arte.dataelab.base_slopes import BaseSlopes


class BaseSlopesTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range10x6.fits')
        self.testdata = fits.getdata(self.fitsfile)

    def test_get_display_remaps_to_slopes_vector_to_2d(self):
        mapper2d = lambda x: x.reshape((3,1))
        series = BaseSlopes(1*u.s, self.testdata*u.m, mapper2d=mapper2d, astropy_unit=None)
        assert series.get_display().shape == (10, 3, 2)


if __name__ == "__main__":
    unittest.main()
