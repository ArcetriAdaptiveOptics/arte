'''
Created on 16 mar 2020

@author: giuliacarla
'''

import unittest
import numpy as np
from apposto.utils.integrator import SimpleIntegrator, IdealIntegrator


class TestIntegrator(unittest.TestCase):

    def testIdealIntegratorValues(self):
        ig = IdealIntegrator()
        rtf_want = 0.
        ntf_want = -1
        rtf_got = ig.getRejectionTransferFunction()
        ntf_got = ig.getNoiseTransferFunction()
        self.assertEqual(rtf_got, rtf_want)
        self.assertEqual(ntf_got, ntf_want)

    def testDifferenceBetweenRTFandNTF(self):
        delay = 3.
        gain = 0.3
        freqs = np.logspace(-3, 3, 500)
        ig = SimpleIntegrator()
        ig.setDelay(delay)
        ig.setGain(gain)
        ig.setTemporalFrequencies(freqs)
        rtf = ig.getRejectionTransferFunction()
        ntf = ig.getNoiseTransferFunction()
        got_real = rtf.real - ntf.real
        got_imag = rtf.imag - ntf.imag
        want_real = np.ones(got_real.shape)
        want_imag = np.zeros(got_imag.shape)
        np.testing.assert_array_almost_equal(got_real, want_real)
        np.testing.assert_array_almost_equal(got_imag, want_imag)
