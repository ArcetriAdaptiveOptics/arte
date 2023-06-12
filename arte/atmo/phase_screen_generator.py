import numpy as np
from arte.utils.discrete_fourier_transform \
    import BidimensionalFourierTransform as bfft


class PhaseScreenGenerator(object):

    def __init__(self,
                 screenSizeInPixels,
                 screenSizeInMeters,
                 outerScaleInMeters,
                 seed=None):
        self._screenSzInPx = screenSizeInPixels
        self._screenSzInM = float(screenSizeInMeters)
        self._outerScaleInM = float(outerScaleInMeters)
        self._phaseScreens = None
        self._nSubHarmonicsToUse = 8
        if seed is None:
            self._seed = np.random.randint(2**32 - 1, dtype=np.uint32)
        else:
            self._seed = seed

    def _spatial_frequency(self, screenSizeInPixels):
        a = np.tile(np.fft.fftfreq(screenSizeInPixels, d=1. / screenSizeInPixels),
                    (screenSizeInPixels, 1))
        return np.linalg.norm(np.dstack((a, a.T)), axis=2)

    def generate_normalized_phase_screens(self, numberOfScreens):
        np.random.seed(self._seed)
        nIters = int(np.ceil(numberOfScreens / 2))
        ret = np.zeros((2 * nIters, self._screenSzInPx, self._screenSzInPx))
        for i in range(nIters):
            ps = self._generate_phase_screen_with_fft()
            ps += self._generate_sub_harmonics(self._nSubHarmonicsToUse)
            ret[2* i, :, :]= self._remove_piston(np.sqrt(2)* ps.real)
            ret[2* i + 1, :, :]= self._remove_piston(np.sqrt(2)* ps.imag)
        self._phaseScreens= ret[:numberOfScreens]


    def _remove_piston(self, scrn):
        return scrn-scrn.mean()


    def _generate_phase_screen_with_fft(self):
        '''
        Normalized to 1pix/r0
        '''
        freqMap = self._spatial_frequency(self._screenSzInPx)
        modul = self._kolmogorov_amplitude_map_no_piston(freqMap)
        phaseScreen = np.sqrt(0.0228) * self._screenSzInPx**(5. / 6) * \
            np.fft.fft2(modul * np.exp(self._random_phase() * 1j))
        return phaseScreen

    def rescale_to(self, r0At500nm):
        self._normalizationFactor = \
            ((self._screenSzInM / self._screenSzInPx) / r0At500nm) ** (5. / 6)

    def get_in_radians_at(self, wavelengthInMeters):
        return 500e-9 / wavelengthInMeters * \
            self._normalizationFactor * self._phaseScreens

    def get_in_meters(self):
        return self._normalizationFactor * self._phaseScreens / \
            (2 * np.pi) * 500e-9

    def _random_phase(self):
        return np.random.rand(self._screenSzInPx, self._screenSzInPx) * 2 * np.pi

    def _kolmogorov_amplitude_map_no_piston(self, freqMap):
        mappa = (freqMap**2 + (self._screenSzInM / self._outerScaleInM)**2)**(
            -11. / 12)
        mappa[0, 0] = 0
        return mappa

    def _generate_sub_harmonics(self, numberOfSubHarmonics):
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
            modul = self._kolmogorov_amplitude_map_no_piston(freqMod)
            for ix in range(nSub):
                for jx in range(nSub):
                    sh = np.exp(2 * np.pi * 1j *
                                (xx * freqX[ix, jx] + yy * freqY[ix, jx] + phase[ix, jx]))
                    sh0 = sh.sum() / self._screenSzInPx**2
                    lowFreqScreen += 1. / nSub**depth * \
                        modul[ix, jx] * (sh - sh0)
        lowFreqScreen *= np.sqrt(0.0228) * self._screenSzInPx**(5. / 6)
        return lowFreqScreen

    def _determineNumberOfSubHarmonics(self):
        maxNoOfSubHarm = 8
        pass
