import numpy as np
from arte.misc.fourier_adaptive_optics import TurbulentPhase, \
    FourierAdaptiveOptics
from arte.utils.discrete_fourier_transform import \
    BidimensionalFourierTransform as bfft
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.modal_decomposer import ModalDecomposer
from arte.types.wavefront import Wavefront
from scipy.special import gamma


def r0AtLambda(r0At500, wavelenghtInMeters):
    return r0At500 * (wavelenghtInMeters / 0.5e-6) ** (6. / 5)


class Test1():

    def __init__(self, dPupInMeters, r0At500nm, wavelenghtInMeters,
                 dPupInPixels=1024, outerScaleInMeter=1e6):
        self._pupDiameterInMeters = dPupInMeters
        self.r0 = r0At500nm
        self._lambda = wavelenghtInMeters
        self._pupDiameterInPixels = dPupInPixels
        self._L0 = outerScaleInMeter

        self._tb = TurbulentPhase()
        self._pxSize = self._pupDiameterInMeters / self._pupDiameterInPixels
        self._spat_freqs = bfft.frequencies_norm_map(self._pupDiameterInPixels,
                                                     self._pxSize)
        self._dist = bfft.distances_norm_map(self._pupDiameterInPixels,
                                             self._pxSize)
        self._mapCenter = (np.asarray(self._dist.shape) / 2).astype(np.int)
        self._psd = self._tb.vonKarmanPowerSpectralDensity(self.r0, self._L0,
                                                           self._spat_freqs)
        self._phaseAC = bfft.direct_transform(self._psd).real
        self._phaseSTF = 2 * (
            self._phaseAC[self._mapCenter[0], self._mapCenter[1]] -
            self._phaseAC)
        self._kolmSTFInRad2 = 6.88 * (
            self._dist / r0AtLambda(self.r0, self._lambda))**(5 / 3)
        # TODO mezzi pixels!


class TestLongExposure():

    def __init__(self, dPupInMeters,
                 howMany=100, dPupInPixels=128, outerScaleInMeters=1e6):
        self.pupil_diameter = dPupInMeters
        self._L0 = outerScaleInMeters
        self._howMany = howMany
        self._nPx = dPupInPixels
        self._psg = PhaseScreenGenerator(
            self._nPx, self.pupil_diameter, self._L0)
        self._psg.generate_normalized_phase_screens(self._howMany)

    def _doPsf(self, phaseScreenInRadians):
        self._fao.setPhaseMapInMeters(
            phaseScreenInRadians / (2 * np.pi) * self._lambda)
        return self._fao.psf().values

    def test(self, r0At500nm, wavelenghtInMeters):
        self.r0 = r0At500nm
        self._lambda = wavelenghtInMeters
        extFact = 2
        self._fao = FourierAdaptiveOptics(
            pupilDiameterInMeters=self.pupil_diameter,
            wavelength=self._lambda,
            resolutionFactor=extFact)
        self._psg.rescale_to(self.r0)
        ps = self._psg.get_in_radians_at(self._lambda)
        aa = np.asarray([self._doPsf(ps[i]) for i in range(self._howMany)])
        longExposurePsf = aa.mean(axis=0)
        freqsX = self._fao.focalPlaneCoordsInArcsec()
        return longExposurePsf, freqsX

    def seeingLimitedFWHMInArcsec(self):
        return 0.976 * self._lambda / \
            r0AtLambda(self.r0, self._lambda) / 4.848e-6

    @staticmethod
    def run():
        from arte.utils.image_moments import ImageMoments
        import matplotlib
        tle = TestLongExposure(8.0, howMany=1000, dPupInPixels=128,
                               outerScaleInMeters=1e6)
        le = tle.test(0.1, 0.5e-6)
        sz = le[0].shape[0]
        matplotlib.pyplot.plot(le[1], le[0][sz // 2, :])
        print(ImageMoments(le[0]).semiAxes() *
              le[1][sz // 2 - 1:sz // 2 + 1][1])
        print(tle.seeingLimitedFWHMInArcsec())
        return tle, le


class AtmosphericPhaseScreenDecomposition():

    def __init__(self):
        self.n_modes = 1275
        self.pupil_diameter = 8
        self.r0 = 0.2
        self.wavelength = 0.5e-6
        self.L0 = 1e12
        self.n_pixel = 64
        self.how_many = 100

        self._md = ModalDecomposer(self.n_modes)
        self._psg = PhaseScreenGenerator(
            self.n_pixel, self.pupil_diameter, self.L0)

        self.compute()

    def compute(self):
        self.generate()
        self.decompose()

    def generate(self):
        self._psg.generate_normalized_phase_screens(self.how_many)
        mask = CircularMask((self.n_pixel, self.n_pixel))
        self._psg.rescale_to(self.r0)
        self._screen_cube = np.ma.masked_array(
            self._psg.get_in_radians_at(self.wavelength),
            np.tile(mask.mask(), (self.how_many, 1, 1)))

    def decompose(self):
        self._zcoeff = np.zeros((self.how_many, self.n_modes))
        mask = CircularMask.fromMaskedArray(self._screen_cube[0])
        for i in range(self.how_many):
            wf = Wavefront.fromNumpyArray(self._screen_cube[i].data)
            zcoeff = self._md.measureZernikeCoefficientsFromWavefront(
                wf, mask)
            self._zcoeff[i] = zcoeff.toNumpyArray()
        self._zIdx = zcoeff.zernikeIndexes()

    def modal_variance(self):
        return self._zcoeff.var(axis=0)

    def expected_modal_variance(self):
        return self.zern_var(self._zIdx) * (self.pupil_diameter / self.r0) ** (5 / 3)

    def zern_var(self, j):
        n = ZernikeGenerator.radialOrder(j)
        return 0.7554 * (n + 1) * gamma(n + 1 - 11 / 6) / gamma(n + 1 + 17 / 6)

    def plot_std(self):
        import matplotlib.pyplot as plt
        plt.loglog(self._zIdx, self.modal_variance(), '.-')
        plt.loglog(self._zIdx, self.expected_modal_variance())
        plt.xlabel("Zernike mode")
        plt.ylabel("Mode variance [rad**2]")
        plt.show()

# class SimulationOfResidualPhase():
#
#     def __init__(self, phase_screen,
#                  layer_altitude,
#                  wind_xy,
#                  source1,
#                  source2,
#                  time_delay,
#                  gain):
#         self._phaseScreen = phase_screen
#         self._layerAlt = layer_altitude
#         self._wind = wind_xy
#         self._source1 = source1
#         self._source2 = source2
#         self._d = time_delay
#         self._g = gain
#         self._source1Coords = self._quantitiesToValue(
#             source1.getSourcePolarCoords())
#         self._source2Coords = self._quantitiesToValue(
#             source2.getSourcePolarCoords())
#
#     def _getPhaseScreenOnAxisAtOneStep(self, step):
#         shift_x = self._wind[step][1]
#         shift_y = self._wind[step][0]
#         ps_x = np.roll(self._phaseScreen, shift_x, axis=1)
#         ps_xy = np.roll(ps_x, shift_y, axis=0)
#         return ps_xy
#
#     def _getPhaseScreenOffAxisAtOneStep(self, step):
#         shift_x = self._layerProjectedAperturesSeparation()[0]
#         shift_y = self._layerProjectedAperturesSeparation()[1]
#         ps_on = self._getPhaseScreenOnAxisAtOneStep(step)
#         ps_off_x = np.roll(ps_on, shift_x, axis=1)
#         ps_off_xy = np.roll(ps_off_x, shift_y, axis=0)
#         return ps_off_xy
#
#     def getPhaseScreenOnAxis(self, n_step):
#         phase_screen_list = []
#         for t in range(n_step):
#             ps = self._getPhaseScreenOnAxisAtOneStep(t)
#             phase_screen_list.append(ps)
#         return phase_screen_list
#
#     def getPhaseScreenOffAxis(self, n_step):
#         phase_screen_list = []
#         for t in range(n_step):
#             ps = self._getPhaseScreenOffAxisAtOneStep(t)
#             phase_screen_list.append(ps)
#         return phase_screen_list
#
#     def _quantitiesToValue(self, quantity):
#         if type(quantity) == list:
#             value = np.array([quantity[i].value for i in range(len(quantity))])
#         else:
#             value = quantity.value
#         return value
#
#     def _getCleverVersorCoords(self, s_coord):
#         return np.array([
#             np.sin(np.deg2rad(s_coord[0] / 3600)) *
#             np.cos(np.deg2rad(s_coord[1])),
#             np.sin(np.deg2rad(s_coord[0] / 3600)) *
#             np.sin(np.deg2rad(s_coord[1])),
#             np.cos(np.deg2rad(s_coord[0] / 3600))])
#
#     def _layerProjectedAperturesSeparation(self):
#         vCoord1 = self._getCleverVersorCoords(self._source1Coords)
#         vCoord2 = self._getCleverVersorCoords(self._source2Coords)
#         sep = self._layerAlt * vCoord2 / vCoord2[2] - \
#             self._layerAlt * vCoord1 / vCoord1[2]
#         return sep
#
#     def _getPhaseOfDMUntilStepN(self, ):
#         pass



    
    
    # Parsevals theorem with proper sample points
    
    #energy_t = np.sum(abs(f)**2, x=t)
    #energy_f = np.trapz(abs(FFT)**2, x=frq) / N
    
    #print('Parsevals theorem NOT fulfilled: ' + str(energy_t - energy_f))
