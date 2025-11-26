import numpy as np
from arte.atmo.abstract_phase_generator import PhaseGenerator
from astropy.io import fits

from typing import override

class PhaseScreenGenerator(PhaseGenerator):

    def __init__(self,
                 screenSizeInPixels,
                 screenSizeInMeters,
                 outerScaleInMeters,
                 nSubHarmonics:int=8,
                 seed:int=None):
        super().__init__(screenSizeInPixels,screenSizeInMeters,nSubHarmonics,seed)
        self._outerScaleInM = float(outerScaleInMeters)

    @override
    def _get_power_spectral_density(self, freqMap):
        mappa = (freqMap**2 + (self._screenSzInM / self._outerScaleInM)**2)**(
            -11. / 12)
        mappa[0, 0] = 0
        return mappa
    
    @override
    def _get_scaling(self):
        return np.sqrt(0.0228) * self._screenSzInPx**(5. / 6)

    def rescale_to(self, r0At500nm):
        self._normalizationFactor = \
            ((self._screenSzInM / self._screenSzInPx) / r0At500nm) ** (5. / 6)

    def get_in_radians_at(self, wavelengthInMeters):
        return 500e-9 / wavelengthInMeters * \
            self._normalizationFactor * self._phaseScreens

    def get_in_meters(self):
        return self._normalizationFactor * self._phaseScreens / \
            (2 * np.pi) * 500e-9
    
    def save_normalized_phase_screens(self, fname, overwrite:bool=False):
        hdr = fits.Header()
        hdr['SZ_IN_PX'] =  self._screenSzInPx
        hdr['SZ_IN_M'] = self._screenSzInM
        hdr['OS_IN_M'] = self._outerScaleInM
        hdr['SEED'] = self._seed
        hdr['NSUBHARM'] = self._nSubHarmonicsToUse
        fits.writeto(fname, self._phaseScreens, hdr, overwrite=overwrite)
        
    @staticmethod 
    def load_normalized_phase_screens(fname):
        header = fits.getheader(fname)
        screenSizeInPixels = header['SZ_IN_PX'] 
        screenSizeInMeters = header['SZ_IN_M'] 
        outerScaleInMeters = header['OS_IN_M'] 
        seed = header['SEED'] 
        try: # added for retro-compatibility
            nSubHarmonics = header['NSUBHARM']
        except KeyError:
            nSubHarmonics = 8
        psg = PhaseScreenGenerator(
            screenSizeInPixels,
            screenSizeInMeters,
            outerScaleInMeters,
            nSubHarmonics,
            seed)
        hduList = fits.open(fname)
        psg._phaseScreens = hduList[0].data
        return psg
