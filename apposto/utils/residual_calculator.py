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

    def getSCAOResidual(self, temporal_freqs):
        scao_res = np.trapz(self._computeSCAOIntegrand(), temporal_freqs)
        return scao_res

    def getMCAOResidual(self, proj_matrix_on, proj_matrix_off,
                        reconstructor, temporal_freq):
        mcao_res = np.trapz(self._computeMCAOIntegrand(proj_matrix_on,
                                                       proj_matrix_off,
                                                       reconstructor,
                                                       temporal_freq),
                            temporal_freq,
                            axis=2)
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
        m_on = np.dot(p_on, w)
        m_off = np.dot(p_off, w)
        id_1 = np.identity(m_off.shape[0])

        integ_list = []
        for i in range(f.shape[0]):
            a_on = np.dot((g * z[i]**(-d) * m_on), np.linalg.inv(
                (1 - z[i]**(-1)) * id_1 + g * z[i]**(-d) * m_off))
            a_off = np.dot((g * z[i]**(-d) * m_off), np.linalg.inv(
                (1 - z[i]**(-1)) * id_1 + g * z[i]**(-d) * m_off))
            id_2 = np.identity(a_off.shape[0])

            integrand = (1 + np.linalg.norm(a_on)**2) * \
                self._cpsd_onon[:, :, i] - \
                2 * np.real(np.dot(a_on, self._cpsd_offon[:, :, i])) + \
                np.linalg.norm(
                    g * z[i]**(-d) / (1 - z[i]**(-1)) * np.dot(
                        m_on, (id_2 - a_off)
                    ))**2 * self._psd_noise[:, :, i]
            integ_list.append(integrand)
        return np.dstack(integ_list)
