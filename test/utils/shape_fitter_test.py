#!/usr/bin/env python
import doctest
import unittest
import numpy as np
from arte.utils.shape_fitter import ShapeFitter
from arte.types.mask import CircularMask, AnnularMask


class PupilFitterTest(unittest.TestCase):

    def setUp(self):
        self.shape = (256, 256)
        self.radius = 52.3
        self.cx = 120.6
        self.cy = 70.9
        self.inradius = 15
        self.params2check_circle = np.array([
            self.cx - 0.5, self.cy - 0.5, self.radius])
        self.params2check_anular = np.array([
            self.cx - 0.5, self.cy - 0.5, self.radius, self.inradius])
        self.testMask1 = CircularMask(self.shape,
                                      maskRadius=self.radius,
                                      maskCenter=(self.cx, self.cy))
        self.testMask2 = AnnularMask(self.shape,
                                     maskRadius=self.radius,
                                     maskCenter=(self.cx, self.cy),
                                     inRadius=self.inradius)

    def test_docstring(self):
        print("In method %s" % self._testMethodName)
        import arte.utils.shape_fitter as pf_module
        doctest.testmod(pf_module, raise_on_error=True)

    def test_ransac(self):
        print("In method %s" % self._testMethodName)
        ff = ShapeFitter(self.testMask1.asTransmissionValue())
        ff.fit_circle_ransac()
        np.testing.assert_allclose(
            ff.parameters(), self.params2check_circle, rtol=0.01)

    def test_circle_correlation(self):
        print("In method %s" % self._testMethodName)
        mm = ['Nelder-Mead']
        for xx in mm:
            print("Tested method %s" % xx)
            ff = ShapeFitter(self.testMask1.asTransmissionValue())
            ff.fit_circle_correlation(
                method=xx, options={'disp': True})
            p1 = ff.parameters()
            print(p1)
            np.testing.assert_allclose(p1, self.params2check_circle, rtol=0.01)

    def test_annular_correlation(self):
        print("In method %s" % self._testMethodName)
        mm = ['Nelder-Mead']
        for xx in mm:
            print("Tested method %s" % xx)
            ff2 = ShapeFitter(self.testMask2.asTransmissionValue())
            ff2.fit_annular_correlation(
                method=xx, options={'disp': True})
            p2 = ff2.parameters()
            print(p2)

#           if xx == 'CG' or xx == 'BFGS' or xx == 'L-BFGS-B' or xx == 'TNC' \
#                     or xx == 'SLSQP' or xx == 'trust-constr':
#                 p1 = p1[[1, 0, 2]]
#                 p2 = p2[[1, 0, 2]]
#                 rtol = 3

            np.testing.assert_allclose(p2, self.params2check_anular, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
