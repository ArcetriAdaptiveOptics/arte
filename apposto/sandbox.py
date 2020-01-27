import numpy as np
from apposto.misc.fourier_adaptive_optics import TurbulentPhase, \
    FourierAdaptiveOptics
from apposto.utils.discrete_fourier_transform import \
    BidimensionalFourierTransform as bfft
from apposto.atmo.phase_screen_generator import PhaseScreenGenerator
from apposto.types.mask import CircularMask


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


class ResidualVarianceUsingOneOffAxisNGS():

    def __init__(self, rejection_TF, noise_TF, turbulence_PSD,
                 turbulence_CrossPSD, noise_PSD):
        self._RTF = rejection_TF
        self._NTF = noise_TF
        self._turbPSD = turbulence_PSD
        self._turbCrossPSD = turbulence_CrossPSD
        self._noisePSD = noise_PSD

    def _getIntegrand(self):
        return self._RTF**2 * self._turbPSD + self._NTF**2 * self._noisePSD + \
            2 * np.real(self._NTF * (self._turbCrossPSD - self._turbPSD))

    def _computeIntegral(self, freqs):
        return np.trapz(self._getIntegrand(), freqs)

    def getResidualVariance(self, temporal_frequencies):
        return self._computeIntegral(temporal_frequencies)


class TransferFunction():

    def __init__(self, temporal_frequencies, gain, temporal_delay):
        self._freqs = temporal_frequencies
        self._gain = gain
        self._delay = temporal_delay
        self._z = np.exp(1j * 2 * np.pi * temporal_frequencies)

    def rejectionTransferFunction(self):
        return (1 - self._z**(-1)) / (1 - self._z**(-1) +
                                      self._gain * self._z**(-self._delay))

    def noiseTransferFunction(self):
        return - (self._gain * self._z**(-self._delay)) / (
            1 - self._z**(-1) + self._gain * self._z**(-self._delay))


# def getOffAxisErrorVsDistance(distances, j, k):
#     from apposto.types.guide_source import GuideSource
#     from apposto.types.aperture import CircularOpticalAperture
#     from apposto.atmo.cn2_profile import Cn2Profile
#     from apposto.atmo.von_karman_covariance_calculator import \
#         VonKarmanSpatioTemporalCovariance
#
#     ngs = GuideSource((0, 0), np.inf)
#     aperture = CircularOpticalAperture(5, [0, 0, 0])
#     cn2 = Cn2Profile.from_r0s([0.16], [25], [10e3], [10], [-20])
#     freqs = np.logspace(-3, 3, 100)
#
#     vk = VonKarmanSpatioTemporalCovariance(source1=ngs, source2=ngs,
#                                            aperture1=aperture,
#                                            aperture2=aperture,
#                                            cn2_profile=cn2,
#                                            spat_freqs=freqs)
#     covOnOn = (vk.getZernikeCovariance(j, k).value).real
#
#     def computeOffAxisError(covOnOn, covOnOff):
#         return 2 * (covOnOn - covOnOff)
#
#     residuals = []
#     for pos in distances:
#         source = GuideSource(pos, np.inf)
#         vk.setSource2(source)
#         cov = (vk.getZernikeCovariance(j, k).value).real
#         res = computeOffAxisError(covOnOn, cov)
#         residuals.append(res)
#     return residuals
#
#
# def ellipsePlot(distances_xy, width, height, angle,
#                 x_max, x_min, y_max, y_min):
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Ellipse
#     ellipse = [Ellipse((distances_xy[i, 0], distances_xy[i, 1]),
#                        width[i], height[i], angle=angle[i]) for i in range(
#         len(width))]
#     fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
#     for e in ellipse:
#         ax.add_artist(e)
#         e.set_clip_box(ax.bbox)
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
#         plt.show()
#
#
# def tipTiltMapExample():
#     import matplotlib.pyplot as plt
#
#     def getCartesianPositions(rho, theta):
#         x = rho * np.cos(np.deg2rad(theta))
#         y = rho * np.sin(np.deg2rad(theta))
#         return x, y
#
#     def computeEllipseParameters(res22, res33, res23):
#         covMatrix = np.array([[res22, res23], [res23, res33]])
#         lam, v = np.linalg.eigh(covMatrix)
#         order = lam.argsort()[::-1]
#         lamSort, vSort = lam[order], v[:, order]
#         theta = np.rad2deg(np.arctan2(vSort[:, 0][1], vSort[:, 0][0]))
#         width, height = 2 * np.sqrt(lamSort)
#         return width, height, theta
#
#     rhos = np.random.randint(-50, 50, size=(500))
#     thetas = np.random.randint(360, size=(500))
#     positionsInPolar = np.stack((rhos, thetas), axis=-1)
#
#     res22 = getOffAxisErrorVsDistance(positionsInPolar, 2, 2)
#     res33 = getOffAxisErrorVsDistance(positionsInPolar, 3, 3)
#     res23 = getOffAxisErrorVsDistance(positionsInPolar, 2, 3)
#
#     positionsInXY = np.array([getCartesianPositions(rhos[i], thetas[i])
#                               for i in range(rhos.shape[0])])
#
#     widths = []
#     heights = []
#     ths = []
#     for i in range(len(res22)):
#         w, h, t = computeEllipseParameters(res22[i], res33[i], res23[i])
#         widths.append(w / 2)
#         heights.append(h / 2)
#         ths.append(t)
#
#     ellipsePlot(distances_xy=positionsInXY, width=widths,
#                 height=heights, angle=ths,
#                 x_max=50, x_min=-50, y_max=50, y_min=-50)
#
#     plt.scatter(0, 0, marker='*', s=100, color='y')
#     plt.xlabel('arcsec')
#     plt.ylabel('arcsec')
