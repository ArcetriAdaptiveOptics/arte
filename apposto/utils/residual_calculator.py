'''
Created on 13 mar 2020

@author: giuliacarla
'''

import numpy as np


class ResidualCalculator():
    """
    """

    def __init__(self, cpsd_onon, cpsd_offon, psd_noise,
                 integrator):
        self._cpsd_onon = cpsd_onon
        self._cpsd_offon = cpsd_offon
        self._psd_noise = psd_noise
        self._integrator = integrator

    def getSCAOResidual(self, temporal_freq):
        scao_res = np.trapz(self._computeSCAOIntegrand(), temporal_freq)
        return scao_res

    def getMCAOResidual(self, proj_matrix_on, proj_matrix_off,
                        reconstructor, temporal_freq):
        mcao_res = np.trapz(self._computeMCAOIntegrand(proj_matrix_on,
                                                       proj_matrix_off,
                                                       reconstructor),
                            temporal_freq)
        return mcao_res

    def _computeSCAOIntegrand(self):
        rtf = self._integrator.getRejectionTransferFunction()
        ntf = self._integrator.getNoiseTransferFunction()
        integrand = rtf**2 * self._cpsd_onon + ntf**2 * self._psd_noise + \
            2 * np.real(ntf * (self._cpsd_offon - self._cpsd_onon))
        return integrand

    def _computeMCAOIntegrand(self, p_on, p_off, w, f):
        d = self._integrator.delay
        g = self._integrator.gain
        z = np.exp(2j * np.pi * f)
        a_on = (g * z**(-d) * p_on * w) / (
            1 - z**(-1) + g * z**(-d) * p_off * w)
        a_off = (g * z**(-d) * p_off * w) / (
            1 - z**(-1) + g * z**(-d) * p_off * w)
        id_matrix = np.identity(a_off.shape[0])

        integrand = (1 + np.linalg.norm(a_on)**2) * self._cpsd_onon - \
            2 * np.real(a_on * self._cpsd_offon) + \
            np.linalg.norm(g * z**(-d) / (1 - z**(-1)) * p_on * w * (
                id_matrix - a_off))**2 * self._psd_noise
# TODO: check the expression
        return integrand
