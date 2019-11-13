'''
@author: giuliacarla
'''

import numpy as np
import matplotlib.pyplot as plt


class VonKarmannPsd():
    '''
    This class computes the spatial Power Spectral Density (PSD) of turbulent
    phase assuming the Von Karmann spectrum.
    The PSD is obtained from the following expression:

        PSD(f,h) = 0.023*r0(h)**(-5/3)**(f**2+1/L0**2)**(-11/6)


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

#     def _getOuterScaleInMeters(self):
#         return self._cn2._layersL0
#
#     def _getFriedParametersInMeters(self):
#         return self._cn2._jsToR0(self._cn2._layersJs,
#                                  self._cn2.airmass(),
#                                  self._cn2.wavelength())
#
#     def _getLayersAltitudeInMeters(self):
#         return self._cn2.layersDistance()

    def _computeVonKarmannPsd(self, freqs):
        if type(self._r0) == np.ndarray:
            self._psd = np.array([0.023 * self._r0[i]**(-5. / 3) *
                                  (freqs**2 + 1 / self._L0**2)**(-11. / 6)
                                  for i in range(self._r0.shape[0])])
        else:
            self._psd = 0.023 * self._r0**(-5. / 3) * \
                (freqs**2 + 1 / self._L0**2)**(-11. / 6)

    def getVonKarmannPsd(self, freqs):
        self._computeVonKarmannPsd(freqs)
        return self._psd

    def plotVonKarmannPsdVsFrequency(self, freqs, idx=None):
        psd = self.getVonKarmannPsd(freqs)
        if idx is None:
            plt.loglog(freqs, psd)
        else:
            plt.loglog(freqs, psd[idx])
        plt.xlabel('Frequency [m$^{-1}$]')
        plt.ylabel('PSD [m$^{2}$]')
