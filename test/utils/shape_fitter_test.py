#!/usr/bin/env python
import doctest
import unittest
import numpy as np
from arte.utils.shape_fitter import ShapeFitter
from arte.types.mask import CircularMask, AnnularMask


class PupilFitterTest(unittest.TestCase):

    def setUp(self):
        self.shape = (256, 256)
        self.radius = 52
        self.cx = 120.5
        self.cy = 70.5
        self.inradius = 15
        self.params2check = np.array([
            self.cx - 0.5, self.cy - 0.5, self.radius])
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
        ff2 = ShapeFitter(self.testMask2.asTransmissionValue())
        ff2.fit_circle_ransac()
        np.testing.assert_allclose(
            ff.parameters(), self.params2check, rtol=1e-2)
        np.testing.assert_allclose(ff2.parameters(),
                                   self.params2check, rtol=1e-2)

    def test_correlation_full(self):
        print("In method %s" % self._testMethodName)
        mm = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
              'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
        for xx in mm:
            print("Tested method %s" % xx)
            ff = ShapeFitter(self.testMask1.asTransmissionValue())
            ff.fit_circle_correlation(method=xx)
            ff2 = ShapeFitter(self.testMask2.asTransmissionValue())
            ff2.fit_circle_correlation(method=xx)
            p1 = ff.parameters()
            p2 = ff2.parameters()
            rtol = 0.01
            if xx == 'CG' or xx == 'BFGS' or xx == 'L-BFGS-B' or xx == 'TNC' \
                    or xx == 'SLSQP' or xx == 'trust-constr':
                p1 = p1[[1, 0, 2]]
                p2 = p2[[1, 0, 2]]
                rtol = 3

            np.testing.assert_allclose(p1, self.params2check, rtol=rtol)
            np.testing.assert_allclose(p2, self.params2check, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
