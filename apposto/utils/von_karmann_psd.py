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
    cn2_profile: cn2 profile as obtained from the Cn2Profile class
            (e.g. cn2_profile = apposto.atmo.cn2_profile.EsoEltProfiles.Q1())
    '''
    
    def __init__(self, cn2_profile):
        self._cn2 = cn2_profile
        
    def _getOuterScaleInMeters(self):
        return self._cn2._layersL0
    
    def _getFriedParametersInMeters(self):
        return self._cn2._jsToR0(self._cn2._layersJs,
                                 self._cn2.airmass(),
                                 self._cn2.wavelength())
    
    def _getLayersAltitudeInMeters(self):
        return self._cn2.layersDistance()
    
    def getVonKarmannPsdOfSingleLayer(self, layer_idx, freqs):
        L0 = self._getOuterScaleInMeters()
        r0s = self._getFriedParametersInMeters()
        return 0.023 * r0s[layer_idx]**(-5./3) * (freqs**2 + 1/L0**2)**(-11./6)
    
    def getVonKarmannPsdOfAllLayers(self, freqs):
        numberOfLayers = self._getLayersAltitudeInMeters().shape[0]
        return np.array([
                self.getVonKarmannPsdOfSingleLayer(i, freqs)
                for i in range(numberOfLayers)])
        
    def plotVonKarmannPsdVsFrequency(self, idx, freqs):
        psd = self.getVonKarmannPsdOfSingleLayer(idx, freqs)
        plt.loglog(freqs, psd)
        plt.xlabel('Frequency')
        plt.ylabel('PSD')