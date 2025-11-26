import numpy as np
from arte.utils.discrete_fourier_transform \
    import BidimensionalFourierTransform as bfft

from abc import ABC, abstractmethod

class PhaseGenerator(ABC):
    """
    Abstract base class for phase screen generators

    Example use:
    >>> class MyPhaseGenerator(PhaseGenerator):
    ...     def _get_power_spectral_density(self, freqMap):
    ...         return np.exp(-freqMap)
    ...
    ...     def _get_scaling(self):
    ...         return 1.0
    >>> phs = MyPhaseGenerator(screenSizeInPixels=256,
    ...                        screenSizeInMeters=10.0,
    ...                        seed=42)
    >>> phs.generate_normalized_phase_screens(numberOfScreens=5)
    >>> phaseScreens = phs._phaseScreens  # access generated screens
    """

    def __init__(self,
                 screenSizeInPixels,
                 screenSizeInMeters,
                 seed:int=None,
                 nSubHarmonics:int=8):
        self._screenSzInPx = screenSizeInPixels
        self._screenSzInM = float(screenSizeInMeters)
        self._phaseScreens = None
        self._nSubHarmonicsToUse = nSubHarmonics
        if seed is None:
            self._seed = np.random.randint(2**32 - 1, dtype=np.uint32)
        else:
            self._seed = seed

    @abstractmethod
    def _get_power_spectral_density(self, freqMap, **kwargs):
        """ Override with the function defining
        the desired power spectral density """
    
    def _get_scaling(self, **kwargs):
        """ Override if needed, to get the correct
        scaling given the pixel size """
        return 1.0

    def _spatial_frequency(self, screenSizeInPixels):
        a = np.tile(np.fft.fftfreq(screenSizeInPixels, d=1. / screenSizeInPixels),
                    (screenSizeInPixels, 1))
        return np.linalg.norm(np.dstack((a, a.T)), axis=2)

    def generate_normalized_phase_screens(self, numberOfScreens, **kwargs):
        np.random.seed(self._seed)
        nIters = int(np.ceil(numberOfScreens / 2))
        ret = np.zeros((2 * nIters, self._screenSzInPx, self._screenSzInPx))
        for i in range(nIters):
            ps = self._generate_phase_screen_with_fft(**kwargs)
            ps += self._generate_sub_harmonics(self._nSubHarmonicsToUse,**kwargs)
            ret[2* i, :, :]= self._remove_piston(np.sqrt(2)* ps.real)
            ret[2* i + 1, :, :]= self._remove_piston(np.sqrt(2)* ps.imag)
        self._phaseScreens= ret[:numberOfScreens]

    def _remove_piston(self, scrn):
        return scrn-scrn.mean()

    def _generate_phase_screen_with_fft(self, **kwargs):
        ''' Normalized to 1pix/r0 '''
        freqMap = self._spatial_frequency(self._screenSzInPx)
        modul = self._get_power_spectral_density(freqMap,**kwargs)
        phaseScreen = np.fft.fft2(modul * np.exp(self._random_phase() * 1j))
        phaseScreen *= self._get_scaling(**kwargs)
        return phaseScreen

    def _random_phase(self):
        return np.random.rand(self._screenSzInPx, self._screenSzInPx) * 2 * np.pi

    def _generate_sub_harmonics(self, numberOfSubHarmonics, **kwargs):
        nSub = 3
        lowFreqScreen = np.zeros((self._screenSzInPx, self._screenSzInPx),
                                 dtype=np.complex128)
        freqX = bfft.frequencies_x_map(nSub, 1. / nSub)
        freqY = bfft.frequencies_y_map(nSub, 1. / nSub)
        freqMod = bfft.frequencies_norm_map(nSub, 1. / nSub)
        vv = np.arange(self._screenSzInPx) / self._screenSzInPx
        xx = np.tile(vv, (self._screenSzInPx, 1))
        yy = xx.T
        depth = 0
        while depth < numberOfSubHarmonics:
            depth += 1
            phase = self._random_phase()
            freqMod /= nSub
            freqX /= nSub
            freqY /= nSub
            modul = self._get_power_spectral_density(freqMod)
            for ix in range(nSub):
                for jx in range(nSub):
                    sh = np.exp(2 * np.pi * 1j *
                                (xx * freqX[ix, jx] + yy * freqY[ix, jx] + phase[ix, jx]))
                    sh0 = sh.sum() / self._screenSzInPx**2
                    lowFreqScreen += 1. / nSub**depth * \
                        modul[ix, jx] * (sh - sh0)
        lowFreqScreen *= self._get_scaling(**kwargs)
        return lowFreqScreen