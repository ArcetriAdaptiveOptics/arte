#!/usr/bin/env python
import unittest
import numpy as np
from arte.utils.marechal import wavefront_rms_2_strehl_ratio, \
    strehl_ratio_2_wavefront_rms, scale_strehl_ratio
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose


class Test(unittest.TestCase):

    def test_wf_2_sr(self):
        got = wavefront_rms_2_strehl_ratio(100 * u.nm, 1 * u.um)
        want = np.exp(-(2 * np.pi * 0.1) ** 2)
        self.assertAlmostEqual(want, got)

    def test_sr_2_wf(self):
        want = 100 * u.nm
        sr = np.exp(-(2 * np.pi * 0.1) ** 2)
        got = strehl_ratio_2_wavefront_rms(sr, 1 * u.um)
        assert_quantity_allclose(want, got)

    def test_scale_sr(self):
        sr = 0.8
        wl1 = 1e-6
        wl2 = 2e-6
        want = 0.8 ** ((wl1 / wl2) ** 2)
        got = scale_strehl_ratio(sr, wl1, wl2)
        assert_quantity_allclose(want, got)
        got2 = scale_strehl_ratio(sr, wl1 * u.m, wl2 * u.m)
        assert_quantity_allclose(want, got2)
        got3 = scale_strehl_ratio(sr, wl1, wl2 * u.m)
        assert_quantity_allclose(want, got3)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
