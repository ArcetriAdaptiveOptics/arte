'''
Created on 16 mar 2020

@author: giuliacarla
'''

import unittest
import numpy as np
from arte.control.transfer_function import ZetaTransferFunction, \
    IdealTransferFunction, LaplaceTransferFunction


class TestIdealTransferFunction(unittest.TestCase):

    def testIdealControlValues(self):
        ic = IdealTransferFunction()
        rtf_want = 0.
        ntf_want = -1
        rtf_got = ic.RTF()
        ntf_got = ic.NTF()
        self.assertEqual(rtf_got, rtf_want)
        self.assertEqual(ntf_got, ntf_want)


class TestZetaTransferFunction(unittest.TestCase):

    def testDifferenceBetweenRTFandNTF(self):
        delay = 3.
        gain = 0.3
        floop = 500
        n_iter = 1e4
        zc = ZetaTransferFunction(floop, n_iter, gain, delay)
        rtf = zc.RTF()
        ntf = zc.NTF()
        got_real = rtf.real - ntf.real
        got_imag = rtf.imag - ntf.imag
        want_real = np.ones(got_real.shape)
        want_imag = np.zeros(got_imag.shape)
        np.testing.assert_array_almost_equal(got_real, want_real)
        np.testing.assert_array_almost_equal(got_imag, want_imag)


class TestLaplaceTransferFunction(unittest.TestCase):

    def testOptimalGain(self):
        f = np.logspace(-3, 3, 5000)
        lp = LaplaceTransferFunction(temporal_freqs=f, gain=1, t_integration=0.001)
        lp.set_optimal_gain()
        oltf = lp.OLTF()
        idx135 = lp.idx_omega_135
        oltf135 = lp.get_amplitude(oltf)[idx135]
        self.assertAlmostEqual(oltf135, 1)

    def testCachedValuesAreUpdated(self):
        f = np.logspace(-3, 3, 5000)
        lp = LaplaceTransferFunction(temporal_freqs=f, gain=1, t_integration=0.001)
        rtf_before = lp.RTF()
        lp.set_optimal_gain()
        rtf_after = lp.RTF()
        self.assertFalse(np.allclose(rtf_before, rtf_after))
        lp.set_integrator(True)
        rtf_last = lp.RTF()
        self.assertFalse(np.allclose(rtf_last, rtf_after))

    def testModifyGainToModifyBandwidth(self):
        f = np.logspace(-3, 3, 5000)
        lp = LaplaceTransferFunction(temporal_freqs=f, gain=1, t_integration=0.001)
        lp.gain = 1
        rbw_before = lp.rejection_bandwidth()
        lp.gain = 2
        rbw_after = lp.rejection_bandwidth()
        self.assertFalse(np.allclose(rbw_before, rbw_after))

