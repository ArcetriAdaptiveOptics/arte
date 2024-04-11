#!/usr/bin/env python

import unittest
import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.kl_generator import KLGenerator
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.types.mask import CircularMask
from numpy.testing import assert_allclose

class TestKLGenerator(unittest.TestCase):

    def setUp(self):
        self._nmodes = 100
        self._nPixels = 101
        self._mask = CircularMask((self._nPixels, self._nPixels), 30, (55,60) )
        self._kolm_covar =getFullKolmogorovCovarianceMatrix(self._nmodes)
        self.generator = KLGenerator(self._mask, covariance_matrix=self._kolm_covar)

    def testRadius(self):
        radius = self.generator.radius()
        self.assertEqual(radius, 30)

    def testCenter(self):
        center = self.generator.center()
        self.assertEqual(center, (55, 60))

    def testGenerateFromBase(self):
        zz = ZernikeGenerator(self._mask)
        zbase = np.rollaxis(np.ma.masked_array([zz.getZernike(n) 
                                    for n in range(2,self._nmodes+2)]),0,3)
        self.generator.generateFromBase(zbase)
        self.assertEqual(self.generator._klBase.shape, (self._nPixels, self._nPixels, self._nmodes))


    def testGetCube(self):
        zz = ZernikeGenerator(self._mask)
        zbase = np.rollaxis(np.ma.masked_array([zz.getZernike(n) 
                                                for n in range(2,self._nmodes+2)]),0,3)
        self.generator.generateFromBase(zbase)
        klcube = self.generator.getCube()
        self.assertEqual(klcube.shape, (self._nPixels, self._nPixels, self._nmodes))

    def testGetKLDict(self):
        zz = ZernikeGenerator(self._mask)
        zbase = np.rollaxis(np.ma.masked_array([zz.getZernike(n) 
                                                for n in range(2,self._nmodes+2)]),0,3)
        self.generator.generateFromBase(zbase)
        kl_dict = self.generator.getKLDict(list(range(4, self._nmodes,2)))
        self.assertEqual(len(kl_dict), (self._nmodes-4)//2)


if __name__ == "__main__":
    unittest.main()
