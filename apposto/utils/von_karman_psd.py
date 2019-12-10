'''
@author: giuliacarla
'''

import numpy as np
from scipy.special import gamma


class VonKarmanPsd():
    '''
    This class computes the spatial Power Spectral Density (PSD) of turbulent
    phase assuming the Von Karman spectrum.
    The PSD is obtained from the following expression:

        PSD(f,h) = (24/5 * gamma(6/5))**(5/6) *
            gamma(11/6)**2 / (2 * np.pi ** (11 / 3)) *
            r0(h)**(-5/3) * (f**2+1/L0**2)**(-11/6)

    Parameters
    ----------
    fried_param: float
        Fried parameter characterizing the atmospheric turbulence [m].
    outer_scale: float
        Outer scale of the atmospheric turbulence [m].

    Example
    -------
    Compute the total variance of Kolmogorov over a 10m telescope:
    R = 5
    r0 = 0.1
    L0 = np.inf
    psd = von_karmann_psd.VonKarmanPsd(r0, L0)
    freqs = np.logspace(-5, 4, 1000)
    bess = scipy.special.jv(1, 2*np.pi*R*freqs)
    psdPistonRem = psd.spatial_psd(freqs) * (1 - (bess/(np.pi*R*freqs))**2)
    varInSquareRad = np.trapz(psdPistonRem*2*np.pi*freqs, freqs)

    Compare varInSquareRad with Noll('76): delta1 = 1.029*(2*R/r0)**(5./3)
    '''

    def __init__(self, fried_param, outer_scale):
        self._r0 = fried_param
        self._L0 = outer_scale
        self._checkInput()

    def _checkInput(self):
        if np.isscalar(self._r0):
            assert np.isscalar(self._L0), 'L0 must be scalar like r0'
        else:
            assert len(self._r0) == len(self._L0), \
                'len of L0 and r0 differ: %d vs %d' % (
                    len(self._L0), len(self._r0))

    def _computeVonKarmanPsd(self, freqs):
        c = (24. / 5 * gamma(6. / 5)) ** (5. / 6) * \
            gamma(11. / 6) ** 2 / (2 * np.pi ** (11 / 3))
        if type(self._r0) == np.ndarray:
            self._psd = np.array([
                c * self._r0[i] ** (-5. / 3) *
                (freqs ** 2 + 1 / self._L0 ** 2) ** (-11. / 6)
                for i in range(self._r0.shape[0])])
        else:
            self._psd = 0.0229 * self._r0 ** (-5. / 3) * \
                (freqs ** 2 + 1 / self._L0 ** 2) ** (-11. / 6)

    def spatial_psd(self, freqs):
        self._computeVonKarmanPsd(freqs)
        return self._psd

    def plot_von_karman_psd_vs_frequency(self, freqs, idx=None):
        import matplotlib.pyplot as plt
        psd = self.spatial_psd(freqs)
        if idx is None:
            plt.loglog(freqs, psd)
        else:
            plt.loglog(freqs, psd[idx])
        plt.xlabel('Frequency [m$^{-1}$]')
        plt.ylabel('PSD [m$^{2}$]')
