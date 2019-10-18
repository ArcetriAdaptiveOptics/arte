#!/usr/bin/env python
import unittest
from astropy import units as u
from apposto.wfs.sh_lenslet_parameters import ShackHartmannLensletParameters



class Test(unittest.TestCase):


    def testName(self):
        dTele= 39*u.m
        nSub= 80
        ccdSize= 1100
        ccdPixelSize= 9*u.micron
        pixelscale= 1.2*u.arcsec

        shp= ShackHartmannLensletParameters.fromSensorSize(
            dTele, nSub, ccdSize, ccdPixelSize, pixelscale)

        self.assertEqual(12, shp.n_pixel_per_subaperture())
        self.assertEqual(
            ccdPixelSize*nSub*shp.n_pixel_per_subaperture(),
            shp.pupil_size_on_lenslet_array())



if __name__ == "__main__":
    unittest.main()
