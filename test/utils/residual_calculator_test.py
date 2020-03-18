'''
Created on 16 mar 2020

@author: giuliacarla
'''

import unittest
from apposto.utils.residual_calculator import ResidualCalculator
from apposto.utils.integrator import SimpleIntegrator
import numpy as np


class TestResidualCalculator(unittest.TestCase):

    def testShapeOfMCAOResidual(self):
        n_modes = 2
        m_modes_layer = 5
        n_NGS = 2
        n_freqs = 100
        cpsd_on = np.ones((n_modes, n_modes, n_freqs))
        cpsd_off = np.ones((n_modes * n_NGS, n_modes, n_freqs))
        cpsd_noise = np.ones((n_modes, n_modes, n_freqs))
        p_on = np.ones((n_modes, m_modes_layer))
        p_off = np.ones((n_modes * n_NGS, m_modes_layer))
        w = np.ones((m_modes_layer, n_modes * n_NGS))
        d = 3.
        g = 0.3
        f = np.logspace(-3, 3, n_freqs)
        ig = SimpleIntegrator()
        ig.setDelay(d)
        ig.setGain(g)
        ig.setTemporalFrequencies(f)

        rc = ResidualCalculator(cpsd_on, cpsd_off, cpsd_noise, ig)
        mcao_res = rc.getMCAOResidual(p_on, p_off, w, f)
        got_shape = mcao_res.shape
        want_shape = (n_modes, n_modes)
        self.assertEqual(got_shape, want_shape)
