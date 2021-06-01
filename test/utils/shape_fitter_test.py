#!/usr/bin/env python
import doctest
import unittest
import numpy as np
from arte.utils.shape_fitter import ShapeFitter
from arte.types.mask import CircularMask, AnnularMask


class PupilFitterTest(unittest.TestCase):

    def setUp(self):
        self.shape = (128, 128)
        self.radius = 52
        self.cx = 60.5
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
        import arte.utils.shape_fitter as pf_module
        doctest.testmod(pf_module, raise_on_error=True)

    def test_ransac(self):
        ff = ShapeFitter(self.testMask1.asTransmissionValue())
        ff.fit(method='RANSAC')
        ff2 = ShapeFitter(self.testMask2.asTransmissionValue())
        ff2.fit(method='RANSAC')
        np.testing.assert_allclose(ff.parameters(), self.params2check, rtol=1e-2)
        np.testing.assert_allclose(ff2.parameters(), self.params2check, rtol=1e-2)

    def test_correlation(self):
        ff = ShapeFitter(self.testMask1.asTransmissionValue())
        ff.fit()
        ff2 = ShapeFitter(self.testMask2.asTransmissionValue())
        ff2.fit()
        np.testing.assert_allclose(ff.parameters(), self.params2check, rtol=1e-2)
        np.testing.assert_allclose(ff2.parameters(), self.params2check, rtol=1e-2)

    def test_correlation_passes_but_it_is_wrong(self):
        want = [1, 2, 3]
        got = [2, 1, 3]
        self.assertEqual(np.round(want, 1).all(),
                         np.round(got, 1).all())

    def test_exceptions(self):

        ff = ShapeFitter(self.testMask1.asTransmissionValue())

        with self.assertRaises(Exception):
            _ = ff.fit(shape2fit='ellipse')

        with self.assertRaises(ValueError):
            _ = ff.fit(method='new')


if __name__ == "__main__":
    unittest.main()
