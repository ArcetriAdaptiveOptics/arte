'''
@author: giuliacarla
'''

import numpy as np
try:
    import cupy as cp
except ImportError:
    print("Can't import cupy")
    cp = None
import astropy.units as u
from arte.utils import math
import logging
from arte.utils.zernike_generator import ZernikeGenerator
from arte.atmo import von_karman_psd
# from arte.utils.decorator import cacheResult


class VonKarmanSpatioTemporalCovariance():
    """
    Covariance of Von Karman atmospheric turbulence

    This class computes the covariance and its Fourier transform, the Cross
    Power Spectral Density (CPSD), between Zernike coefficients describing
    the turbulence induced phase aberrations of two sources seen by two
    different circular apertures. The CPSD of the phase is also computed.


    References
    ----------
        Plantet et al. (2020) - "Spatio-temporal statistics of the turbulent
        Zernike coefficients and piston-removed phase from two distinct beams".

        Plantet et al. (2018) - "LO WFS of MAORY: performance and sky coverage
        assessment."

        Whiteley et al. (1998) - "Temporal properties of the Zernike expansion
        coefficients of turbulence-induced phase aberrations for aperture
        and source motion".


    Parameters
    ----------
    source1: `~arte.types.guide_source.GuideSource`
        Geometry of the first source.
        We consider rho as the angle in arcsec wrt the z-axis and
        theta as the angle in degrees wrt the x-axis.
        (e.g. source1 = arte.types.guide_source.GuideSource((1,90), 9e3)

    source2: `~arte.types.guide_source.GuideSource`
        Geometry of the second source. Same conventions as source1.

    aperture1: `~arte.types.aperture.CircularOpticalAperture`
        Geometry of the first optical aperture.
        (e.g. aperture1 = arte.types.aperture.CircularOpticalAperture(
        10, (0, 0, 0)))

    aperture2: `~arte.types.aperture.CircularOpticalAperture`
        Geometry of the second optical aperture.

    cn2_profile: `~arte.atmo.cn2_profile`
        Cn2 profile.
        (e.g. cn2_eso = arte.atmo.cn2_profile.EsoEltProfiles.Q1()
        e.g. cn2_invented = arte.atmo.cn2_profile.Cn2Profile.from_r0s(
        [0.16], [25], [10e3], [0.1], [0]))

    spat_freqs: `numpy.ndarray`
        Range of spatial frequencies that are used in Zernike covariance,
        Zernike CPSD and phase CPSD computation.

    """

    def __init__(self,
                 source1,
                 source2,
                 aperture1,
                 aperture2,
                 cn2_profile,
                 spat_freqs,
                 logger=logging.getLogger('VK_COVARIANCE')):
        self.setSource1(source1)
        self.setSource2(source2)
        self.setAperture1(aperture1)
        self.setAperture2(aperture2)
        self.setCn2Profile(cn2_profile)
        self.setSpatialFrequencies(spat_freqs)
        self._xp = np
        self._logger = logger

    def useGPU(self):
        self._xp = cp

    def setCn2Profile(self, cn2_profile):
        self._resetCachedResults()
        self._cn2 = cn2_profile
        self._layersDistance = self._quantitiesToValue(
            cn2_profile.layers_distance())
        self._r0s = self._quantitiesToValue(cn2_profile.r0s())
        self._outerScale = self._quantitiesToValue(cn2_profile.outer_scale())
        self._numberOfLayers = self._quantitiesToValue(
            cn2_profile.number_of_layers())
        self._windSpeed = self._quantitiesToValue(cn2_profile.wind_speed())
        self._windDirection = self._quantitiesToValue(
            cn2_profile.wind_direction())

    def setSource1(self, s1):
        self._resetCachedResults()
        self._source1 = s1
        self._source1Coords = self._quantitiesToValue(
            s1.getSourcePolarCoords())

    def setSource2(self, s2):
        self._resetCachedResults()
        self._source2 = s2
        self._source2Coords = self._quantitiesToValue(
            s2.getSourcePolarCoords())

    def setAperture1(self, ap1):
        self._resetCachedResults()
        self._ap1 = ap1
        self._ap1Coords = self._quantitiesToValue(
            ap1.getCartesianCoords())
        self._ap1Radius = self._quantitiesToValue(
            ap1.getApertureRadius())

    def setAperture2(self, ap2):
        self._resetCachedResults()
        self._ap2 = ap2
        self._ap2Coords = self._quantitiesToValue(
            ap2.getCartesianCoords())
        self._ap2Radius = self._quantitiesToValue(
            ap2.getApertureRadius())

    def setSpatialFrequencies(self, freqs):
        self._resetCachedResults()
        self._spat_freqs = freqs

    def _resetCachedResults(self):
        # TODO: lb 200620. This is ugly, the decorator should offer
        # the method to reset itself. It could also mean that the
        # methods setSource, setAperture etc are not needed and one
        # should instantiate a new object instead of modifying it.
        # Also, note that we call this method to reset everything
        # every time something is modified, and this is
        # likely suboptimal
        self._initializeGeometry_cached_result = {}

    def source1(self):
        return self._source1

    def source2(self):
        return self._source2

    def aperture1(self):
        return self._ap1

    def aperture2(self):
        return self._ap2

    def cn2_profile(self):
        return self._cn2

    def spatial_frequencies(self):
        return self._spat_freqs

    def temporal_frequencies(self):
        return self._temporalFreqs

    def _quantitiesToValue(self, quantity):
        if type(quantity) == list:
            value = np.array([quantity[i].value for i in range(len(quantity))])
        else:
            value = quantity.value
        return value

    def _layerScalingFactor1(self, nLayer):
        a1 = self._layerScalingFactor(
            self._layersDistance[nLayer],
            self._source1Coords[2],
            self._ap1Coords[2])
        return a1

    def _layerScalingFactor2(self, nLayer):
        a2 = self._layerScalingFactor(
            self._layersDistance[nLayer],
            self._source2Coords[2],
            self._ap2Coords[2])
        return a2

    def _layerScalingFactor(self, zLayer, zSource, zAperture):
        scalFact = (zLayer - zAperture) / (zSource - zAperture)
        return scalFact

    def _getCleverVersorCoords(self, s_coord):
        return np.array([
            np.sin(np.deg2rad(s_coord[0] / 3600)) *
            np.cos(np.deg2rad(s_coord[1])),
            np.sin(np.deg2rad(s_coord[0] / 3600)) *
            np.sin(np.deg2rad(s_coord[1])),
            np.cos(np.deg2rad(s_coord[0] / 3600))])

    def _cleverVersor1Coords(self):
        return self._getCleverVersorCoords(
            self._source1Coords)

    def _cleverVersor2Coords(self):
        return self._getCleverVersorCoords(
            self._source2Coords)

    def _layerProjectedAperturesSeparation(self, nLayer):
        aCoord1 = self._ap1Coords
        aCoord2 = self._ap2Coords

        vCoord1 = self._cleverVersor1Coords()
        vCoord2 = self._cleverVersor2Coords()

        sep = aCoord2 - aCoord1 + (
            self._layersDistance[nLayer] - aCoord2[2]) * \
            vCoord2 / vCoord2[2] - \
            (self._layersDistance[nLayer] - aCoord1[2]) * \
            vCoord1 / vCoord1[2]
        return sep

    def _VonKarmanPSDOneLayer(self, nLayer, freqs):
        vk = von_karman_psd.VonKarmanPsd(
            self._r0s[nLayer], self._outerScale[nLayer])
        psd = vk.spatial_psd(freqs)
        return psd

#     @cacheResult
    def _initializeGeometry(self, nLayer, spat_freqs):
        a1 = self._layerScalingFactor1(nLayer)
        a2 = self._layerScalingFactor2(nLayer)
        sep = self._xp.array(self._layerProjectedAperturesSeparation(nLayer))

        if self._xp == cp:
            psd = self._VonKarmanPSDOneLayer(
                nLayer, self._xp.asnumpy(spat_freqs))
            psd = self._xp.array(psd)
        elif self._xp == np:
            psd = self._VonKarmanPSDOneLayer(nLayer, spat_freqs)
        sl = self._xp.linalg.norm(sep)
        thS = self._xp.arctan2(sep[1], sep[0])
        return a1, a2, sl, thS, psd

    def _initializeFunctionsForZernikeComputations(
            self, j, k, spat_freqs, a1, a2):
        nj, mj = ZernikeGenerator.degree(j)
        nk, mk = ZernikeGenerator.degree(k)

        deltaj = math.kroneckerDelta(0, mj)
        deltak = math.kroneckerDelta(0, mk)

        if self._xp == cp:
            r1 = cp.array(self._ap1Radius)
            r2 = cp.array(self._ap2Radius)
            b1 = math.besselFirstKindOnGPU(
                nj + 1,
                2 * self._xp.pi * spat_freqs * r1 * (1 - a1))
            b2 = math.besselFirstKindOnGPU(
                nk + 1,
                2 * self._xp.pi * spat_freqs * r2 * (1 - a2))
        elif self._xp == np:
            r1 = self._ap1Radius
            r2 = self._ap2Radius
            b1 = math.besselFirstKind(
                nj + 1,
                2 * self._xp.pi * spat_freqs * r1 * (1 - a1))
            b2 = math.besselFirstKind(
                nk + 1,
                2 * self._xp.pi * spat_freqs * r2 * (1 - a2))

        c0 = (-1) ** mk * self._xp.sqrt((nj + 1) * (nk + 1)) * 1j ** (
            nj + nk) * 2 ** (
            1 - 0.5 * (deltaj + deltak))
        c1 = 1. / (
            self._xp.pi * r1 * r2 * (1 - a1) * (1 - a2))

        return nj, mj, nk, mk, deltaj, deltak, b1, b2, c0, c1

    def _computeZernikeCovarianceOneLayer(self, j, k, nLayer):
        f = self._xp.array(self._spat_freqs)
        a1, a2, sl, thS, psd = \
            self._initializeGeometry(nLayer, f)
        _, mj, _, mk, deltaj, deltak, b1, b2, c0, c1 = \
            self._initializeFunctionsForZernikeComputations(
                j, k, f, a1, a2)

        if self._xp == cp:
            b3 = math.besselFirstKindOnGPU(
                mj + mk,
                sl * 2 * self._xp.pi * f)
            b4 = math.besselFirstKindOnGPU(
                np.abs(mj - mk),
                sl * 2 * self._xp.pi * f)
        elif self._xp == np:
            b3 = math.besselFirstKind(
                mj + mk,
                sl * 2 * self._xp.pi * f)
            b4 = math.besselFirstKind(
                np.abs(mj - mk),
                sl * 2 * self._xp.pi * f)
        c2 = self._xp.pi / 4 * ((1 - deltaj) * ((-1) ** j - 1) +
                                (1 - deltak) * ((-1) ** k - 1))
        c3 = self._xp.pi / 4 * ((1 - deltaj) * ((-1) ** j - 1) -
                                (1 - deltak) * ((-1) ** k - 1))

        zernikeIntegrand = c0 * c1 * \
            psd / f * b1 * b2 * \
            (self._xp.cos((mj + mk) * thS + c2) *
             1j ** (3 * (mj + mk)) *
             b3 +
             np.cos((mj - mk) * thS + c3) *
             1j ** (3 * self._xp.abs(mj - mk)) *
             b4)
        if self._xp == cp:
            zernikeIntegrand = self._xp.asnumpy(zernikeIntegrand)
            f = self._xp.asnumpy(f)
        zernikeCovOneLayer = np.trapezoid(zernikeIntegrand, f)
        return zernikeCovOneLayer

    def _computeZernikeCovarianceAllLayers(self, j, k):
        zernikeCovAllLayers = np.array([
            self._computeZernikeCovarianceOneLayer(j, k, nLayer) for nLayer
            in range(self._numberOfLayers)])
        return zernikeCovAllLayers.sum()

    def _initializeFunctionsForPhaseComputations(self, f, a1, a2):
        if self._xp == cp:
            r1 = cp.array(self._ap1Radius)
            r2 = cp.array(self._ap2Radius)
            k = (1 - a2) * r2 / ((1 - a1) * r1)
            arg1 = self._xp.pi * r1 * (1 - a1)
            arg2 = self._xp.pi * r2 * (1 - a2)

            b0 = math.besselFirstKindOnGPU(1, 2 * arg1 * (k - 1) * f)
#             def b0(f): return math.besselFirstKindOnGPU(
#                 1, 2 * arg1 * (k - 1) * f)

            b1 = math.besselFirstKindOnGPU(1, 2 * arg1 * f)
#             def b1(f): return math.besselFirstKindOnGPU(
#                 1, 2 * arg1 * f)

            b2 = math.besselFirstKindOnGPU(1, 2 * arg2 * f)
#             def b2(f): return math.besselFirstKindOnGPU(
#                 1, 2 * arg2 * f)
        elif self._xp == np:
            k = (1 - a2) * self._ap2Radius / ((1 - a1) * self._ap1Radius)
            arg1 = np.pi * self._ap1Radius * (1 - a1)
            arg2 = np.pi * self._ap2Radius * (1 - a2)
            b0 = math.besselFirstKind(1, 2 * arg1 * (k - 1) * f)
#             def b0(f): return math.besselFirstKind(
#                 1, 2 * arg1 * (k - 1) * f)
            b1 = math.besselFirstKind(1, 2 * arg1 * f)
#             def b1(f): return math.besselFirstKind(
#                 1, 2 * arg1 * f)
            b2 = math.besselFirstKind(1, 2 * arg2 * f)
#             def b2(f): return math.besselFirstKind(
#                 1, 2 * arg2 * f)

        if k == 1:
            #             self._b1Phase = lambda f: f**0
            self._b1Phase = 1
        else:
            #             self._b1Phase = lambda f: b0(
            #                 f) / (arg1 * (k - 1) * f)
            self._b1Phase = b0 / (arg1 * (k - 1) * f)

        if a1 == 1:
            #             self._b2Phase = lambda f: f**0
            self._b2Phase = 1
        else:
            #             self._b2Phase = lambda f: b1(
            #                 f) / (arg1 * f)
            self._b2Phase = b1 / (arg1 * f)

        if a2 == 1:
            #             self._b3Phase = lambda f: f**0
            self._b3Phase = 1
        else:
            #             self._b3Phase = lambda f: b2(
            #                 f) / (arg2 * f)
            self._b3Phase = b2 / (arg2 * f)

    def _computePhaseCovarianceOneLayer(self, nLayer):
        a1, a2, sl, thS, psd = \
            self._initializeGeometry(nLayer, self._spat_freqs)
        self._initializeFunctionsForPhaseComputations(
            self._spat_freqs, a1, a2)
        b0Phase = math.besselFirstKind(0,
                                       2 * np.pi * self._spat_freqs * sl)

        phaseIntegrand = 2 * np.pi * self._spat_freqs * psd * b0Phase * (
            self._b1Phase - self._b2Phase * self._b3Phase)
        phaseCovOneLayer = np.trapezoid(phaseIntegrand, self._spat_freqs)
        return phaseCovOneLayer

    def _getZernikeCPSDAllTemporalFrequenciesOneLayer(self, j, k, nLayer,
                                                      temp_freqs):

        integrand, fPerp = self.integrandOfZernikeCPSD(j, k, nLayer,
                                                       temp_freqs)
        cpsdZernikeOneLayer = np.trapezoid(integrand, fPerp, axis=0)
        return cpsdZernikeOneLayer

    def _getZernikeCPSDAllLayers(self, j, k, temp_freqs):
        cpsd = np.array([
            self._getZernikeCPSDAllTemporalFrequenciesOneLayer(
                j, k, nLayer, temp_freqs) for nLayer
            in range(self._numberOfLayers)])
        cpsdTotal = cpsd.sum(axis=0)
        return cpsdTotal

    def _getGeneralZernikeCPSDAllTemporalFrequenciesOneLayer(self, j, k,
                                                             nLayer,
                                                             temp_freqs):

        integrand, fPerp = self.integrandOfGeneralZernikeCPSD(j, k, nLayer,
                                                              temp_freqs)
        cpsdGeneralZernikeOneLayer = np.trapezoid(integrand, fPerp, axis=0)
        return cpsdGeneralZernikeOneLayer

    def _getGeneralZernikeCPSDAllLayers(self, j, k, temp_freqs):
        cpsd = np.array([
            self._getGeneralZernikeCPSDAllTemporalFrequenciesOneLayer(
                j, k, nLayer, temp_freqs) for nLayer
            in range(self._numberOfLayers)])
        cpsdTotal = cpsd.sum(axis=0)
        return cpsdTotal

    def _getPhaseCPSDAllTemporalFrequenciesOneLayer(self, nLayer, t_freqs):
        func, fPerp = self.integrandOfPhaseCPSD(nLayer, t_freqs)
        cpsdPhaseOneLayer = np.trapezoid(func, fPerp, axis=0)
        return cpsdPhaseOneLayer

    def _getGeneralPhaseCPSDAllTemporalFrequenciesOneLayer(self, nLayer,
                                                           t_freqs):
        func, fPerp = self.integrandOfGeneralPhaseCPSD(nLayer, t_freqs)
        genCpsdPhaseOneLayer = np.trapezoid(func, fPerp, axis=0)
#         inte = self.integrandOfGeneralPhaseCPSD(nLayer, t_freqs)(
#             self._spat_freqs)
#         genCpsdPhaseOneLayer, _ = integrate.quad_vec(inte,
#                                                      self._spat_freqs[0],
#                                                      self._spat_freqs[
#                                                          self._spat_freqs.shape[0]
#                                                          - 1])
#         genCpsdPhaseOneLayer = np.trapz(inte, self._spat_freqs, axis=0)
        return genCpsdPhaseOneLayer

    def _getPhaseCPSDAllLayers(self, temp_freqs):
        phaseCPSD = np.array([
            self._getPhaseCPSDAllTemporalFrequenciesOneLayer(
                nLayer, temp_freqs)
            for nLayer in range(self._numberOfLayers)])
        phaseCPSDTotal = phaseCPSD.sum(axis=0)
        return phaseCPSDTotal

    def _getGeneralPhaseCPSDAllLayers(self, temp_freqs):
        gPhaseCPSD = np.array([
            self._getGeneralPhaseCPSDAllTemporalFrequenciesOneLayer(
                nLayer, temp_freqs)
            for nLayer in range(self._numberOfLayers)])
        gPhaseCPSDTotal = gPhaseCPSD.sum(axis=0)
        return gPhaseCPSDTotal

    def integrandOfGeneralZernikeCPSD(self, j, k, nLayer, temp_freqs):
        if self._xp == cp:
            vl = self._xp.array(self._windSpeed[nLayer])
            vdir = self._xp.array(self._windDirection[nLayer])
            tf, sp_perp = self._xp.meshgrid(
                self._xp.array(temp_freqs), self._xp.array(self._spat_freqs))
        elif self._xp == np:
            vl = self._windSpeed[nLayer]
            vdir = self._windDirection[nLayer]
            tf, sp_perp = self._xp.meshgrid(temp_freqs, self._spat_freqs)

        f = self._xp.sqrt(sp_perp ** 2 + (tf / vl) ** 2)
        a1, a2, sl, thS, psd = self._initializeGeometry(nLayer, f)
        nj, mj, nk, mk, deltaj, deltak, b1, b2, c0, c1 = \
            self._initializeFunctionsForZernikeComputations(
                j, k, f, a1, a2)

        thWind = self._xp.deg2rad(vdir)
        th0 = self._xp.arccos(-tf / (f * vl))
        th1 = th0 + thWind
        th2 = -th0 + thWind

        c4 = self._xp.pi / 4 * (1 - deltaj) * ((-1) ** j - 1)
        c5 = self._xp.pi / 4 * (1 - deltak) * ((-1) ** k - 1)

        integFunc = 2 * c0 * c1 * 1j ** (0.5 * ((-1) ** (
            nj + nk) - 1)) / (vl * self._xp.pi) * \
            psd / f ** 2 * b1 * b2 * \
            (self._xp.cos(2 * self._xp.pi * f * sl * self._xp.cos(th1 - thS) +
                          self._xp.pi / 4 * ((-1) ** (nj + nk) - 1)) *
             self._xp.cos(mj * th1 + c4) *
             self._xp.cos(mk * th1 + c5) +
             self._xp.cos(2 * self._xp.pi * f * sl * self._xp.cos(th2 - thS) +
                          self._xp.pi / 4 * ((-1) ** (nj + nk) - 1)) *
             self._xp.cos(mj * th2 + c4) *
             self._xp.cos(mk * th2 + c5))

        if self._xp == cp:
            return self._xp.asnumpy(integFunc), self._xp.asnumpy(sp_perp)
        elif self._xp == np:
            return integFunc, sp_perp

    def integrandOfZernikeCPSD(self, j, k, nLayer, temp_freqs):
        if self._xp == cp:
            vl = self._xp.array(self._windSpeed[nLayer])
            vdir = self._xp.array(self._windDirection[nLayer])
            tf, sp_perp = self._xp.meshgrid(
                self._xp.array(temp_freqs), self._xp.array(self._spat_freqs))
        elif self._xp == np:
            vl = self._windSpeed[nLayer]
            vdir = self._windDirection[nLayer]
            tf, sp_perp = self._xp.meshgrid(temp_freqs, self._spat_freqs)

        f = self._xp.sqrt(sp_perp ** 2 + (tf / vl) ** 2)
        a1, a2, sl, thS, psd = self._initializeGeometry(nLayer, f)
        _, mj, _, mk, deltaj, deltak, b1, b2, c0, c1 = \
            self._initializeFunctionsForZernikeComputations(
                j, k, f, a1, a2)

        thWind = self._xp.deg2rad(vdir)
        th0 = self._xp.arccos(-tf / (f * vl))
        th1 = th0 + thWind
        th2 = -th0 + thWind

        c4 = self._xp.pi / 4 * (1 - deltaj) * ((-1) ** j - 1)
        c5 = self._xp.pi / 4 * (1 - deltak) * ((-1) ** k - 1)

        integFunc = c0 * c1 / (vl * self._xp.pi) * \
            psd / f ** 2 * b1 * b2 * \
            (self._xp.exp(-2 * 1j * self._xp.pi * f * sl * self._xp.cos(
                th1 - thS)) *
             self._xp.cos(mj * th1 + c4) *
             self._xp.cos(mk * th1 + c5) +
             self._xp.exp(-2 * 1j * self._xp.pi * f * sl * self._xp.cos(
                 th2 - thS)) *
             self._xp.cos(mj * th2 + c4) *
             self._xp.cos(mk * th2 + c5))

        if self._xp == cp:
            return self._xp.asnumpy(integFunc), self._xp.asnumpy(sp_perp)
        elif self._xp == np:
            return integFunc, sp_perp

    def integrandOfPhaseCPSD(self, nLayer, temp_freqs):
        if self._xp == cp:
            vl = self._xp.array(self._windSpeed[nLayer])
            vdir = self._xp.array(self._windDirection[nLayer])
            tf, sp_perp = self._xp.meshgrid(
                self._xp.array(temp_freqs), self._xp.array(self._spat_freqs))
        elif self._xp == np:
            vl = self._windSpeed[nLayer]
            vdir = self._windDirection[nLayer]
            tf, sp_perp = self._xp.meshgrid(temp_freqs, self._spat_freqs)

        f = self._xp.sqrt(sp_perp ** 2 + (tf / vl) ** 2)
        a1, a2, sl, thS, psd = \
            self._initializeGeometry(nLayer, f)
        self._initializeFunctionsForPhaseComputations(f, a1, a2)

        thWind = self._xp.deg2rad(vdir)
        th0 = self._xp.arccos(-tf / (f * vl))
        th1 = th0 + thWind
        th2 = -th0 + thWind

        intFunc = 1. / vl * psd * (
            self._b1Phase - self._b2Phase * self._b3Phase) * (
                self._xp.exp(-2 * 1j * self._xp.pi * f * sl * self._xp.cos(
                    th1 - thS)) +
            self._xp.exp(-2 * 1j * self._xp.pi * f * sl * self._xp.cos(
                th2 - thS)))

        if self._xp == cp:
            return self._xp.asnumpy(intFunc), self._xp.asnumpy(sp_perp)
        elif self._xp == np:
            return intFunc, sp_perp

    def integrandOfGeneralPhaseCPSD(self, nLayer, temp_freqs):
        if self._xp == cp:
            vl = self._xp.array(self._windSpeed[nLayer])
            vdir = self._xp.array(self._windDirection[nLayer])
            tf, sp_perp = self._xp.meshgrid(
                self._xp.array(temp_freqs), self._xp.array(self._spat_freqs))
#             tf = self._xp.array(temp_freqs)
        elif self._xp == np:
            vl = self._windSpeed[nLayer]
            vdir = self._windDirection[nLayer]
            tf, sp_perp = self._xp.meshgrid(temp_freqs, self._spat_freqs)
#             tf = temp_freqs

        f = self._xp.sqrt(sp_perp ** 2 + (tf / vl) ** 2)
#         def f(sf_perp):
#             tf_g, sf_g = np.meshgrid(tf, sf_perp)
#             return self._xp.sqrt(sf_g ** 2 + (tf_g / vl) ** 2)

        a1, a2, sl, thS, psd = \
            self._initializeGeometry(nLayer, f)
        self._initializeFunctionsForPhaseComputations(f, a1, a2)

        thWind = self._xp.deg2rad(vdir)
        th0 = self._xp.arccos(-tf / (f * vl))

#         def th0(f):
#             return self._xp.arccos(-tf / (f * vl))

#         def cos1(f):
#             return f**0 * self._xp.cos(
#                 2 * np.pi * f * sl * self._xp.cos(thWind - thS) *
#                 self._xp.cos(th0(f)))

#         def cos2(f):
#             return f**0 * self._xp.cos(
#                 2 * np.pi * f * sl * self._xp.sin(thWind - thS) *
#                 self._xp.sin(th0(f)))

        intFunc = 4. / vl * psd * (
            self._b1Phase - self._b2Phase * self._b3Phase
        ) * self._xp.cos(
            2 * np.pi * f * sl * self._xp.cos(thWind - thS) * self._xp.cos(
                th0)) * self._xp.cos(
            2 * np.pi * f * sl * self._xp.sin(thWind - thS) *
            self._xp.sin(th0))

        if self._xp == cp:
            return self._xp.asnumpy(intFunc), self._xp.asnumpy(sp_perp)
        elif self._xp == np:
            return intFunc, sp_perp
#         return lambda sf_perp: 4. / vl * self._VonKarmanPSDOneLayer(
#             nLayer, f(sf_perp)) * (
#             self._b1Phase(f(sf_perp)) - self._b2Phase(f(sf_perp)) *
#             self._b3Phase(f(sf_perp))) * \
#             cos1(f(sf_perp)) * cos2(f(sf_perp))

    def getZernikeCovariance(self, j, k):
        """
        Return the covariance between two Zernike coefficients with index j
        and k describing the phase seen, respectively, on aperture1 from
        source1 and on aperture2 from source2.

        Parameters
        ----------
        j: int or list
            Index of Zernike coefficients related to source1 on aperture1.

        k: int or list
            Index of Zernike coefficients related to source2 on aperture2.

        Returns
        -------
        zernikeCovariance: `~astropy.units.quantity.Quantity`
            Zernike covariance or covariance matrix (matrix of shape nxm if n
            and m are, respectively, the dimension of j and k) in [rad**2].
        """

        if (np.isscalar(j) and np.isscalar(k)):
            zernikeCovariance = self._computeZernikeCovarianceAllLayers(j, k)

        elif (np.isscalar(j) and np.isscalar(k) is False):
            zernikeCovariance = np.array([
                self._computeZernikeCovarianceAllLayers(j, k_mode)
                for k_mode in k])

        elif (np.isscalar(j) is False and np.isscalar(k)):
            zernikeCovariance = np.array([
                self._computeZernikeCovarianceAllLayers(j_mode, k)
                for j_mode in j])

        else:
            zernikeCovariance = np.array([
                [self._computeZernikeCovarianceAllLayers(j_mode, k_mode)
                    for k_mode in k]
                for j_mode in j])

        return zernikeCovariance * u.rad ** 2

    def getPhaseCovariance(self):
        """
        Return the covariance between the phase seen from source1 on
        aperture1 and the phase seen from source2 on aperture2.

        Returns
        -------
        phaseCovariance: `~astropy.units.quantity.Quantity`
            Covariance between phase1 and phase2 in [rad**2].
        """
        phaseCovAllLayers = np.array([
            self._computePhaseCovarianceOneLayer(nLayer) for nLayer
            in range(self._numberOfLayers)])
        phaseCovariance = phaseCovAllLayers.sum()
        return phaseCovariance * u.rad ** 2

    def getZernikeCPSD(self, j, k, temp_freqs):
        """
        Return the Cross Power Spectral Density (CPSD) of the Zernike
        coefficients with index j and k describing the phase seen on aperture1
        and aperture2 observing, respectively, source1 and source2.
        The CPSD is a function of the temporal frequency.

        Parameters
        ----------
        j: int
            Index of the Zernike coefficient (related to source1 on aperture1).
        k: int
            Index of the Zernike coefficient (related to source2 on aperture2).
        temp_freqs: numpy.ndarray
            Temporal frequencies array in Hz.

        Returns
        -------
        zernikeCPSD: `~astropy.units.quantity.Quantity`
            Zernike CPSD or matrix of Zernike CPSDs in [rad**2/Hz].
        """

        self._temporalFreqs = temp_freqs

        if (np.isscalar(j) and np.isscalar(k)):
            zernikeCPSD = self._getZernikeCPSDAllLayers(
                j, k, temp_freqs)

        elif (np.isscalar(j) and np.isscalar(k) is False):
            zernikeCPSD = np.array([
                self._getZernikeCPSDAllLayers(j, k_mode, temp_freqs)
                for k_mode in k])

        elif (np.isscalar(j) is False and np.isscalar(k)):
            zernikeCPSD = np.array([
                self._getZernikeCPSDAllLayers(j_mode, k, temp_freqs)
                for j_mode in j])

        else:
            zernikeCPSD = np.array([
                [self._getZernikeCPSDAllLayers(j_mode, k_mode, temp_freqs)
                    for k_mode in k]
                for j_mode in j])

        return zernikeCPSD * u.rad ** 2 / u.Hz

    def getGeneralZernikeCPSD(self, j, k, temp_freqs):
        # TODO: Is this function necessary? We can get the same result
        # computing 2 * np.real(getZernikeCPSD).
        """
        Return the generalized expression of Zernike CPSD that we get
        from 'getZernikeCPSD' function.
        This expression is needed when we have to integrate the Zernike CPSD
        in the temporal frequency range from -infinity to +infinity. Instead of
        this computation, we can obtain the same result performing the integral
        of the generalized Zernike CPSD in the temporal frequency range from
        0 to infinity. This is what we need, for example, when we want to
        compare the Zernike covariance with the Zernike CPSD integrated in
        the temporal frequencies' domain.

        Parameters
        ----------
        j: int or list
            Index of Zernike coefficients related to source1 on aperture1.
        k: int or list
            Index of Zernike coefficients related to source2 on aperture2.
        temp_freqs: numpy.ndarray
            Temporal frequencies array.

        Returns
        -------
        cpsdTotal: `~astropy.units.quantity.Quantity`
            General Zernike CPSD in [rad**2/Hz].
        """

        self._temporalFreqs = temp_freqs

        if (np.isscalar(j) and np.isscalar(k)):
            zernikeCPSD = self._getGeneralZernikeCPSDAllLayers(
                j, k, temp_freqs)

        elif (np.isscalar(j) and np.isscalar(k) is False):
            zernikeCPSD = np.array([
                self._getGeneralZernikeCPSDAllLayers(j, k_mode, temp_freqs)
                for k_mode in k])

        elif (np.isscalar(j) is False and np.isscalar(k)):
            zernikeCPSD = np.array([
                self._getGeneralZernikeCPSDAllLayers(j_mode, k, temp_freqs)
                for j_mode in j])

        else:
            zernikeCPSD = np.array([
                [self._getGeneralZernikeCPSDAllLayers(j_mode, k_mode,
                                                      temp_freqs)
                    for k_mode in k]
                for j_mode in j])

        return zernikeCPSD * u.rad ** 2 / u.Hz

    def getPhaseCPSD(self, temp_freqs):
        """
        Return the Cross Power Spectral Density (CPSD) of the turbulent phase
        seen by aperture1 and aperture2 observing, respectively, source1 and
        source2.
        The CPSD is a function of temporal frequency.

        Parameters
        ----------
        temp_freqs: numpy.ndarray
            Temporal frequencies array.

        Returns
        -------
        phaseCPSD: `~astropy.units.quantity.Quantity`
            Phase CPSD in [rad**2/Hz].
        """

        self._temporalFreqs = temp_freqs

        phaseCPSD = self._getPhaseCPSDAllLayers(temp_freqs) * u.rad ** 2 / u.Hz
        return phaseCPSD

    def getGeneralPhaseCPSD(self, temp_freqs):
        """
        """

        self._temporalFreqs = temp_freqs

        generalPhaseCPSD = self._getGeneralPhaseCPSDAllLayers(
            temp_freqs) * u.rad ** 2 / u.Hz
        return generalPhaseCPSD

    def plotCPSD(self, cpsd, func_part, scale, legend):
        import matplotlib.pyplot as plt
#         if wavelength is None:
#             lam = self._cn2.DEFAULT_LAMBDA
#         else:
#             lam = wavelength
        lam = self._cn2.wavelength().value
        temp_freqs = self.temporal_frequencies()
        m_to_nm = 1e18
        if func_part == 'real':
            if scale == 'log':
                plt.loglog(
                    temp_freqs,
                    np.abs(np.real(cpsd)) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label=legend)
            elif scale == 'linear':
                plt.semilogx(
                    temp_freqs,
                    np.real(cpsd) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label=legend)
        elif func_part == 'imag':
            if scale == 'log':
                plt.loglog(
                    temp_freqs,
                    np.abs(np.imag(cpsd)) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label=legend)
            elif scale == 'linear':
                plt.semilogx(
                    temp_freqs,
                    np.imag(cpsd) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label=legend)
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('CPSD [nm$^{2}$/Hz]')
