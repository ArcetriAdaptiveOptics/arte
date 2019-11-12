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
    cn2_profile: cn2 profile as obtained from the Cn2Profile class
            (e.g. cn2_profile = apposto.atmo.cn2_profile.EsoEltProfiles.Q1())
    '''

    def __init__(self, cn2_profile):
        self._cn2 = cn2_profile

    def _get_outer_scale_in_meters(self):
        return self._cn2.outer_scale()

    def _get_fried_parameters_in_meters(self):
        return self._cn2.r0s()

    def _get_layers_altitude_in_meters(self):
        return self._cn2.layers_distance()

    def spatial_psd_of_single_layer(self, layer_idx, freqs):
        L0 = self._get_outer_scale_in_meters()
        r0s = self._get_fried_parameters_in_meters()
        return 0.0229 * r0s[layer_idx]**(-5. / 3) * \
            (freqs**2 + 1 / L0**2)**(-11. / 6)

    def spatial_psd_of_all_layers(self, freqs):
        number_of_layers = self._get_layers_altitude_in_meters().shape[0]
        return np.array([
            self.spatial_psd_of_single_layer(i, freqs)
            for i in range(number_of_layers)])

    def plot_von_karmann_psd_vs_spatial_frequency(self, idx, freqs):
        import matplotlib.pyplot as plt
        psd = self.spatial_psd_of_single_layer(idx, freqs)
        plt.loglog(freqs, psd)
        plt.xlabel('Frequency [m$^{-1}$]')
        plt.ylabel('PSD [m$^{2}$]')
