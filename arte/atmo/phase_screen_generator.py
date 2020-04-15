import numpy as np
from arte.utils.discrete_fourier_transform \
    import BidimensionalFourierTransform as bfft


class PhaseScreenGenerator(object):

    def __init__(self,
                 screenSizeInPixels,
                 screenSizeInMeters,
                 outerScaleInMeters):
        self._screenSzInPx= screenSizeInPixels
        self._screenSzInM= float(screenSizeInMeters)
        self._outerScaleInM= float(outerScaleInMeters)
        self._phaseScreens= None
        self._nSubHarmonicsToUse= 3
        self._seed= 0


    def _spatialFrequency(self, screenSizeInPixels):
        a=np.tile(np.fft.fftfreq(screenSizeInPixels, d=1./ screenSizeInPixels),
                  (screenSizeInPixels, 1))
        return np.linalg.norm(np.dstack((a, a.T)), axis=2)


    def generateNormalizedPhaseScreens(self, numberOfScreens):
        np.random.seed(self._seed)
        nIters= int(np.ceil(numberOfScreens / 2))
        ret= np.zeros((2* nIters, self._screenSzInPx, self._screenSzInPx))
        for i in range(nIters):
            ps= self._generatePhaseScreenWithFFT()
            ps += self._generateSubHarmonics(self._nSubHarmonicsToUse)
            ret[2* i, :, :]= np.sqrt(2)* ps.real
            ret[2* i + 1, :, :]= np.sqrt(2)* ps.imag
        self._phaseScreens= ret[:numberOfScreens]


    def _generatePhaseScreenWithFFT(self):
        '''
        Normalized to 1pix/r0
        '''
        freqMap= self._spatialFrequency(self._screenSzInPx)
        modul= self._kolmogorovAmplitudeMap(freqMap)
        phaseScreen= np.sqrt(0.0228)* self._screenSzInPx**(5./ 6)* \
            np.fft.fft2(modul * np.exp(self._randomPhase() * 1j))
        return phaseScreen


    def rescaleTo(self, r0At500nm):
        self._normalizationFactor= \
            ((self._screenSzInM/ self._screenSzInPx)/ r0At500nm) ** (5./ 6)


    def getInRadiansAt(self, wavelengthInMeters):
        return 500e-9 / wavelengthInMeters * \
            self._normalizationFactor * self._phaseScreens


    def getInMeters(self):
        return self._normalizationFactor * self._phaseScreens / \
            (2 * np.pi) * 500e-9


    def _randomPhase(self):
        return np.random.rand(self._screenSzInPx, self._screenSzInPx)* 2* np.pi


    def _kolmogorovAmplitudeMap(self, freqMap):
        return (freqMap**2 + (self._screenSzInM / self._outerScaleInM)**2)**(
            -11. / 12)


    def _generateSubHarmonics(self, numberOfSubHarmonics):
        nSub= 3
        lowFreqScreen= np.zeros((self._screenSzInPx, self._screenSzInPx),
                                dtype=np.complex)
        freqX= bfft.frequenciesXMap(nSub, 1./ nSub)
        freqY= bfft.frequenciesYMap(nSub, 1./ nSub)
        freqMod= bfft.frequenciesNormMap(nSub, 1./ nSub)
        vv= np.arange(self._screenSzInPx) / self._screenSzInPx
        xx= np.tile(vv, (self._screenSzInPx, 1))
        yy= xx.T
        depth= 0
        while depth < numberOfSubHarmonics:
            depth+= 1
            phase= self._randomPhase()
            freqMod/= nSub
            freqX/= nSub
            freqY/= nSub
            modul= self._kolmogorovAmplitudeMap(freqMod)
            for ix in range(nSub):
                for jx in range(nSub):
                    sh= np.exp(2* np.pi* 1j*
                        (xx*freqX[ix, jx] + yy*freqY[ix, jx] + phase[ix, jx]))
                    sh0= sh.sum() / self._screenSzInPx**2
                    lowFreqScreen+= 1./ nSub**depth * modul[ix, jx]* (sh- sh0)
        lowFreqScreen*= np.sqrt(0.0228) * self._screenSzInPx**(5. / 6)
        return lowFreqScreen


    def _determineNumberOfSubHarmonics(self):
        maxNoOfSubHarm= 8
        pass
