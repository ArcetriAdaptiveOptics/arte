import numpy as np
from arte.atmo.abstract_phase_screen_generator import AbstractPhaseScreenGenerator
from astropy.io import fits
from arte.utils.decorator import override


class PhaseScreenGenerator(AbstractPhaseScreenGenerator):
    """ 
    Class for atmospheric phase screens generation

    Example use:
    >>> phs = PhaseScreenGenerator(screenSizeInPixels=256,
    ...                            screenSizeInMeters=10.0,
    ...                            outerScaleInMeters=25.0,
    ...                            seed=42)
    >>> phs.generate_normalized_phase_screens(numberOfScreens=5)
    >>> phs.rescale_to(15e-2)  # r0 at 500nm = 15 cm
    >>> phaseScreensInM = phs.get_in_meters()
    >>> phaseScreensAt1um = phs.get_in_radians_at(1e-6)
    """


    def __init__(self,
                 screenSizeInPixels,
                 screenSizeInMeters,
                 outerScaleInMeters,
                 seed:int=None,
                 nSubHarmonics:int=8):
        super().__init__(screenSizeInPixels,screenSizeInMeters,seed,nSubHarmonics)
        self._outerScaleInM = float(outerScaleInMeters)

    @override
    def _get_power_spectral_density(self, freqMap):
        """ Von Karman PSD """
        mappa = (freqMap**2 + (self._screenSzInM / self._outerScaleInM)**2)**(
            -11. / 12)
        mappa[0, 0] = 0
        return mappa
    
    @override
    def _get_scaling(self):
        """ Scaling factor for Von Karman spectrum """
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
    
    def save_normalized_phase_screens(self, filepath:str, overwrite:bool=False):
        hdr = fits.Header()
        hdr['SZ_IN_PX'] =  self._screenSzInPx
        hdr['SZ_IN_M'] = self._screenSzInM
        hdr['OS_IN_M'] = self._outerScaleInM
        hdr['SEED'] = self._seed
        hdr['NSUBHARM'] = self._nSubHarmonicsToUse
        fits.writeto(filepath, self._phaseScreens, hdr, overwrite=overwrite)
        
    @staticmethod 
    def load_normalized_phase_screens(filepath:str):
        header = fits.getheader(filepath)
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
        hduList = fits.open(filepath)
        psg._phaseScreens = hduList[0].data
        return psg
