'''
@author: giuliacarla
'''

import numpy as np


class VonKarmannPsd():
    '''
    This class computes the spatial Power Spectral Density (PSD) of turbulent
    phase assuming the Von Karmann spectrum.
    The PSD is obtained from the following expression:

        PSD(f,h) = 0.0229 * r0(h)**(-5/3) * (f**2+1/L0**2)**(-11/6)

    Parameters
    ----------
    fried_param:
    outer_scale:
    '''

    def __init__(self, fried_param, outer_scale):
        self._r0 = fried_param
        self._L0 = outer_scale

    # TODO: testare la funzione in qualche applicazione (ES:
    #      L0 infinito -> Kolmogorov. Integrale sulle frequenze
    #      mi deve dare la potenza (esistono formuline sulla turbolenza
    #      di Kolomogorov.

    def _computeVonKarmannPsd(self, freqs):
        if type(self._r0) == np.ndarray:
            self._psd = np.array([0.023 * self._r0[i]**(-5. / 3) *
                                  (freqs**2 + 1 / self._L0**2)**(-11. / 6)
                                  for i in range(self._r0.shape[0])])
        else:
            self._psd = 0.0229 * self._r0**(-5. / 3) * \
                (freqs**2 + 1 / self._L0**2)**(-11. / 6)

    def spatial_psd(self, freqs):
        self._computeVonKarmannPsd(freqs)
        return self._psd

    def plot_von_karmann_psd_vs_frequency(self, freqs, idx=None):
        import matplotlib.pyplot as plt
        psd = self.spatial_psd(freqs)
        if idx is None:
            plt.loglog(freqs, psd)
        else:
            plt.loglog(freqs, psd[idx])
        plt.xlabel('Frequency [m$^{-1}$]')
        plt.ylabel('PSD [m$^{2}$]')
