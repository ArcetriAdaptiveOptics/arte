#!/usr/bin/env python
import doctest
import unittest
import numpy as np
from arte.utils.surface_fitter import SurfaceFitter
from arte.utils.zernike_generator import ZernikeGenerator


class SurfaceFitterTest(unittest.TestCase):

    def setUp(self):

        nPix = 1024
        nz = 36
        x0 = 400
        y0 = 600
        r = 300
        self._img = np.zeros((nPix, nPix))
        self._mask = np.ones((nPix, nPix)) == 1

        self._zindex = np.arange(nz) + 1
        self._generator = ZernikeGenerator(2 * r)
        self._set_coeff = np.random.random(size=nz)
        self._mask[x0 - r:x0 + r, y0 - r:y0 + r] = self._generator[1].mask

        for j in np.arange(nz):
            self._img[x0 - r:x0 + r, y0 - r:y0 + r] +=  \
                (self._generator[j + 1].data * self._set_coeff[j])

        self._img = self._img * (self._mask == False)
        self._img_ma = np.ma.masked_array(self._img, self._mask)
        idm = np.where(self._mask == False)

        self._umatg = np.zeros((idm[0].size, nz))
        for j in range(nz):
            self._umatg[:, j] = self._generator[1 + j].compressed()
        self._umat = None
        self._testimg = None
        self._get_coeff = None

    def testZernike(self):
        sft = SurfaceFitter(self._img_ma)
        sft._coords[2] = 300.
        sft.fit(np.arange(1, 37))
        self._get_coeff = sft.coeffs()
        img = self._img * 0
        img[~self._mask] = self._umatg[:, 2]
        mg = self._img * 0
        mg[sft._mask] = sft._umat[:, 2]
        # print(self._set_coeff)
        # print(self._get_coeff - self._set_coeff)
        np.testing.assert_allclose(
            self._set_coeff, self._get_coeff, atol=1e-14)

    def testPoly(self):
        sft = SurfaceFitter(self._img_ma)
        sft._coords[2] = 300.
        sft.fit(np.arange(1, 37), base='poly')
        self._get_coeff = sft.coeffs()
        img = self._img * 0
        img[~self._mask] = self._umatg[:, 12]
        mg = self._img * 0
        mg[sft._mask] = sft._umat[:, 12]
        # print(self._set_coeff)
        # print(self._get_coeff - self._set_coeff)
        #np.testing.assert_allclose(self._set_coeff, self._get_coeff, 1e-14)

    def test_docstring(self):
        import arte.utils.surface_fitter as pf_module
        doctest.testmod(pf_module, raise_on_error=True)

    #
    #
    # def test_exceptions(self):
    #
    #     ff = SurfaceFitter(self.testMask1.asTransmissionValue())
    #
    #     with self.assertRaises(Exception):
    #         _ = ff.fit(shape2fit='ellipse')
    #
    #     with self.assertRaises(ValueError):
    #         _ = ff.fit(method='new')


if __name__ == "__main__":
    unittest.main()
