#!/usr/bin/env python
import unittest
import numpy as np
from apposto.atmo.cn2_profile import Cn2Profile
from apposto.utils.von_karmann_psd import VonKarmannPsd


class VonKarmannPsdTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testValue(self):
        # TODO: I don't like this method
        r0 = 0.1
        L0 = 10
        vk_psd = VonKarmannPsd(r0, L0)
        freqs = np.array([0.1, 1])
        want = 0.0229 * r0**(-5 / 3) * (freqs**2 + 1 / L0**2)**(-11 / 6)
        psd = vk_psd.spatial_psd(freqs)
        self.assertAlmostEqual(want[0], psd[0], delta=.1)
        self.assertAlmostEqual(want[1], psd[1], delta=.1)


if __name__ == "__main__":
    unittest.main()
