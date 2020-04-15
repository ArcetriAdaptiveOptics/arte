#!/usr/bin/env python
import unittest
from arte.photometry.spectral_types import PickelsLibrary
from arte.photometry.spectral_types import O5V, F5I


class PickelsLibraryTest(unittest.TestCase):

    def testFilename(self):
        self.assertEqual(
            'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_27.fits',
            PickelsLibrary.filename('G5V'))

    def testDirectImport(self):
        self.assertEqual(
            'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_1.fits',
            O5V)
        self.assertEqual(
            'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_122.fits',
            F5I)


if __name__ == "__main__":
    unittest.main()
