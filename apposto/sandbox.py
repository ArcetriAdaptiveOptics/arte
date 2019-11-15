import numpy as np
from apposto.misc.fourier_adaptive_optics import TurbulentPhase, FourierAdaptiveOptics
from apposto.utils.discrete_fourier_transform import BidimensionalFourierTransform \
    as bfft
from apposto.atmo.phase_screen_generator import PhaseScreenGenerator
from apposto.types.mask import CircularMask
from apposto.utils.zernike_generator import ZernikeGenerator


def r0AtLambda(r0At500, wavelenghtInMeters):
    return r0At500 * (wavelenghtInMeters / 0.5e-6) ** (6. / 5)


class Test1():

    def __init__(self, dPupInMeters, r0At500nm, wavelenghtInMeters,
                 dPupInPixels=1024, outerScaleInMeter=1e6):
        self._pupDiameterInMeters = dPupInMeters
        self._r0 = r0At500nm
        self._lambda = wavelenghtInMeters
        self._pupDiameterInPixels = dPupInPixels
        self._L0 = outerScaleInMeter

        self._tb = TurbulentPhase()
        self._pxSize = self._pupDiameterInMeters / self._pupDiameterInPixels
        self._freqs = bfft.frequenciesNormMap(self._pupDiameterInPixels,
                                              self._pxSize)
        self._dist = bfft.distancesNormMap(self._pupDiameterInPixels,
                                           self._pxSize)
        self._mapCenter = (np.asarray(self._dist.shape) / 2).astype(np.int)
        self._psd = self._tb.vonKarmanPowerSpectralDensity(self._r0, self._L0,
                                                           self._freqs)
        self._phaseAC = bfft.directTransform(self._psd).real
        self._phaseSTF = 2 * (
            self._phaseAC[self._mapCenter[0], self._mapCenter[1]] -
            self._phaseAC)
        self._kolmSTFInRad2 = 6.88 * (
            self._dist / r0AtLambda(self._r0, self._lambda))**(5 / 3)
        # TODO mezzi pixels!


class TestPhaseScreens():

    def __init__(self, dPupInMeters, r0At500nm, wavelenghtInMeters):
        self._dPup = dPupInMeters
        self._r0 = r0At500nm
        self._lambda = wavelenghtInMeters
        self._L0 = 1e6
        self._howMany = 100
        self._nPx = 128

    def meanStd(self, ps):
        return np.mean(np.std(ps, axis=(1, 2)))

    def stdInRad(self, dTele, r0AtLambda):
        return np.sqrt(1.0299 * (dTele / r0AtLambda)**(5. / 3))

    def test(self):
        psg = PhaseScreenGenerator(self._nPx, self._dPup, self._L0)
        psg.generateNormalizedPhaseScreens(self._howMany)
        mask = CircularMask((self._nPx, self._nPx))
        got = self.meanStd(np.ma.masked_array(
            psg.rescaleTo(self._r0).getInRadiansAt(self._lambda),
            np.tile(mask.mask(), (self._howMany, 1))))
        want = self.stdInRad(self._dPup,
                             r0AtLambda(self._r0, self._lambda))
        print('%g %g %g -> got %g want %g / ratio %f' %
              (self._dPup, self._r0, self._lambda, got, want, want / got))


class TestLongExposure():

    def __init__(self, dPupInMeters,
                 howMany=100, dPupInPixels=128, outerScaleInMeters=1e6):
        self._dPup = dPupInMeters
        self._L0 = outerScaleInMeters
        self._howMany = howMany
        self._nPx = dPupInPixels
        self._psg = PhaseScreenGenerator(self._nPx, self._dPup, self._L0)
        self._psg.generateNormalizedPhaseScreens(self._howMany)

    def _doPsf(self, phaseScreenInRadians):
        self._fao.setPhaseMapInMeters(
            phaseScreenInRadians / (2 * np.pi) * self._lambda)
        return self._fao.psf()

    def test(self, r0At500nm, wavelenghtInMeters):
        self._r0 = r0At500nm
        self._lambda = wavelenghtInMeters
        extFact = 1
        self._fao = FourierAdaptiveOptics(
            self._dPup, self._nPx, self._lambda, extFact)
        ps = self._psg.rescaleTo(self._r0).getInRadiansAt(self._lambda)
        aa = np.asarray([self._doPsf(ps[i]) for i in range(self._howMany)])
        longExposurePsf = aa.mean(axis=0)
        freqsX = self._fao.focalPlaneCoordsInArcsec()
        return longExposurePsf, freqsX

    def seeingLimitedFWHMInArcsec(self):
        return 0.976 * self._lambda / \
            r0AtLambda(self._r0, self._lambda) / 4.848e-6

    @staticmethod
    def run():
        from apposto.utils.image_moments import ImageMoments
        import matplotlib
        tle = TestLongExposure(8.0, howMany=1000, dPupInPixels=128,
                               outerScaleInMeters=1e6)
        le = tle.test(0.1, 2.2e-6)
        matplotlib.pyplot.plot(le[1][64, :], le[0][64, :])
        print(ImageMoments(le[0]).semiAxes() * le[1][64, 64:66][1])
        print(tle.seeingLimitedFWHMInArcsec())
