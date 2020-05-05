#!/usr/bin/env python
import unittest
import numpy as np
from arte.atmo.von_karman_psd import VonKarmanPsd


class VonKarmannPsdTest(unittest.TestCase):

    def testValue(self):
        # TODO: I don't like this method
        r0 = 0.1
        L0 = 10
        vk_psd = VonKarmanPsd(r0, L0)
        freqs = np.array([0.1, 1])
        want = VonKarmanPsd.NUM_CONST * r0 ** (-5 / 3) * (
            freqs ** 2 + 1 / L0 ** 2) ** (-11 / 6)
        psd = vk_psd.spatial_psd(freqs)
        self.assertAlmostEqual(want[0], psd[0], delta=.1)
        self.assertAlmostEqual(want[1], psd[1], delta=.1)

# FIXME! IT SHOULD FAIL
    def testWrongParameters(self):
        r0 = 0.1
        L0 = 10
        self.failUnlessRaises(Exception, VonKarmanPsd, (r0, L0))

    def test_r0_single_element_ndarray(self):
        freqs = np.array([0.1, 1, 10])
        r0 = np.array([0.15])
        L0 = np.array([25])
        psd_sca = VonKarmanPsd(0.15, 25).spatial_psd(freqs)
        psd_arr = VonKarmanPsd(r0, L0).spatial_psd(freqs)
        self.assertTrue(np.allclose(psd_sca, psd_arr[0]),
                        "%s %s" % (psd_sca, psd_arr[0]))

    def test_r0_ndarray(self):
        freqs = np.array([0.1, 1, 10])
        r0 = np.array([0.15, 0.15])
        L0 = np.array([25, 25])
        r0_equiv = np.sum(r0 ** (-5 / 3)) ** (-3 / 5)
        psd_sca = VonKarmanPsd(r0_equiv, 25).spatial_psd(freqs)
        psd_arr = VonKarmanPsd(r0, L0).spatial_psd(freqs)
        self.assertTrue(np.allclose(psd_sca, psd_arr[0] + psd_arr[1]),
                        "%s %s" % (psd_sca, psd_arr[0] + psd_arr[1]))


if __name__ == "__main__":
    unittest.main()
