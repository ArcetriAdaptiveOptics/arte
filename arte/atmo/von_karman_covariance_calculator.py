'''
@author: giuliacarla
'''

import numpy as np
import astropy.units as u
from arte.utils import von_karman_psd, math
import logging
from arte.utils.zernike_generator import ZernikeGenerator


class VonKarmanSpatioTemporalCovariance():
    """
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
        self._source1 = source1
        self._source2 = source2
        self._ap1 = aperture1
        self._ap2 = aperture2
        self._cn2 = cn2_profile
        self._freqs = spat_freqs
        self._logger = logger

        self._source1Coords = self._quantitiesToValue(
            source1.getSourcePolarCoords())
        self._source2Coords = self._quantitiesToValue(
            source2.getSourcePolarCoords())
        self._ap1Coords = self._quantitiesToValue(
            aperture1.getCartesianCoords())
        self._ap2Coords = self._quantitiesToValue(
            aperture2.getCartesianCoords())
        self._ap1Radius = self._quantitiesToValue(
            aperture1.getApertureRadius())
        self._ap2Radius = self._quantitiesToValue(
            aperture2.getApertureRadius())
        self._layersDistance = self._quantitiesToValue(
            cn2_profile.layers_distance())
        self._r0s = self._quantitiesToValue(cn2_profile.r0s())
        self._outerScale = self._quantitiesToValue(cn2_profile.outer_scale())
        self._numberOfLayers = self._quantitiesToValue(
            cn2_profile.number_of_layers())
        self._windSpeed = self._quantitiesToValue(cn2_profile.wind_speed())
        self._windDirection = self._quantitiesToValue(
            cn2_profile.wind_direction())

    def setCn2Profile(self, cn2_profile):
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
        self._source1 = s1
        self._source1Coords = self._quantitiesToValue(
            s1.getSourcePolarCoords())

    def setSource2(self, s2):
        self._source2 = s2
        self._source2Coords = self._quantitiesToValue(
            s2.getSourcePolarCoords())

    def setAperture1(self, ap1):
        self._ap1 = ap1
        self._ap1Coords = self._quantitiesToValue(
            ap1.getCartesianCoords())
        self._ap1Radius = self._quantitiesToValue(
            ap1.getApertureRadius())

    def setAperture2(self, ap2):
        self._ap2 = ap2
        self._ap2Coords = self._quantitiesToValue(
            ap2.getCartesianCoords())
        self._ap2Radius = self._quantitiesToValue(
            ap2.getApertureRadius())

    def setSpatialFrequencies(self, freqs):
        self._freqs = freqs

    def source1(self):
        return self._source1

    def source2(self):
        return self._source2

    def aperture1(self):
        return self._ap1

    def aperture2(self):
        return self._ap2

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

    def _integrate(self, int_func, int_var):
        return np.trapz(np.real(int_func), int_var) + \
            1j * np.trapz(np.imag(int_func), int_var)

    def _initializeGeometry(self, nLayer, spat_freqs):
        self._a1 = self._layerScalingFactor1(nLayer)
        self._a2 = self._layerScalingFactor2(nLayer)
        self._R1 = self._ap1Radius
        self._R2 = self._ap2Radius
        sep = self._layerProjectedAperturesSeparation(nLayer)
        self._sl = np.linalg.norm(sep)
        self._thS = np.arctan2(sep[1], sep[0])
        self._psd = self._VonKarmanPSDOneLayer(nLayer, spat_freqs)

    def _initializeFunctionsForZernikeComputations(self, j, k, spat_freqs):
        self._nj, self._mj = ZernikeGenerator.degree(j)
        self._nk, self._mk = ZernikeGenerator.degree(k)

        self._deltaj = math.kroneckerDelta(0, self._mj)
        self._deltak = math.kroneckerDelta(0, self._mk)

        self._b1 = math.besselFirstKind(
            self._nj + 1,
            2 * np.pi * spat_freqs * self._R1 * (1 - self._a1))
        self._b2 = math.besselFirstKind(
            self._nk + 1,
            2 * np.pi * spat_freqs * self._R2 * (1 - self._a2))

        self._c0 = (-1) ** self._mk * np.sqrt((
            self._nj + 1) * (self._nk + 1)) * 1j ** (
            self._nj + self._nk) * 2 ** (
            1 - 0.5 * (self._deltaj + self._deltak))
        self._c1 = 1. / (
            np.pi * self._R1 * self._R2 * (1 - self._a1) * (1 - self._a2))

    def _computeZernikeCovarianceOneLayer(self, j, k, nLayer):
        self._initializeGeometry(nLayer, self._freqs)
        self._initializeFunctionsForZernikeComputations(j, k, self._freqs)

        self._b3 = math.besselFirstKind(
            self._mj + self._mk,
            self._sl * 2 * np.pi * self._freqs)
        self._b4 = math.besselFirstKind(
            np.abs(self._mj - self._mk),
            self._sl * 2 * np.pi * self._freqs)
        self._c2 = np.pi / 4 * ((1 - self._deltaj) * ((-1) ** j - 1) +
                                (1 - self._deltak) * ((-1) ** k - 1))
        self._c3 = np.pi / 4 * ((1 - self._deltaj) * ((-1) ** j - 1) -
                                (1 - self._deltak) * ((-1) ** k - 1))

        zernikeIntegrand = self._c0 * self._c1 * \
            self._psd / self._freqs * self._b1 * self._b2 * \
            (np.cos((self._mj + self._mk) * self._thS + self._c2) *
             1j ** (3 * (self._mj + self._mk)) *
             self._b3 +
             np.cos((self._mj - self._mk) * self._thS + self._c3) *
             1j ** (3 * np.abs(self._mj - self._mk)) *
             self._b4)
        zernikeCovOneLayer = self._integrate(zernikeIntegrand, self._freqs)
        return zernikeCovOneLayer

    def _computeZernikeCovarianceAllLayers(self, j, k):
        zernikeCovAllLayers = np.array([
            self._computeZernikeCovarianceOneLayer(j, k, nLayer) for nLayer
            in range(self._numberOfLayers)])
        return zernikeCovAllLayers.sum()

    def _initializeFunctionsForPhaseComputations(self, spat_freqs):
        k = (1 - self._a2) * self._R2 / ((1 - self._a1) * self._R1)
        arg1 = np.pi * self._R1 * (1 - self._a1)
        arg2 = np.pi * self._R2 * (1 - self._a2)
        b0 = math.besselFirstKind(1, 2 * arg1 * (k - 1) * spat_freqs)
        b1 = math.besselFirstKind(1, 2 * arg1 * spat_freqs)
        b2 = math.besselFirstKind(1, 2 * arg2 * spat_freqs)

        if k == 1:
            self._b1Phase = 1
        else:
            self._b1Phase = b0 / (arg1 * (k - 1) * spat_freqs)

        if self._a1 == 1:
            self._b2Phase = 1
        else:
            self._b2Phase = b1 / (arg1 * spat_freqs)

        if self._a2 == 1:
            self._b3Phase = 1
        else:
            self._b3Phase = b2 / (arg2 * spat_freqs)

    def _computePhaseCovarianceOneLayer(self, nLayer):
        self._initializeGeometry(nLayer, self._freqs)
        self._initializeFunctionsForPhaseComputations(self._freqs)
        b0Phase = math.besselFirstKind(0,
                                       2 * np.pi * self._freqs * self._sl)

        phaseIntegrand = 2 * np.pi * self._freqs * self._psd * b0Phase * (
            self._b1Phase - self._b2Phase * self._b3Phase)
        phaseCovOneLayer = self._integrate(phaseIntegrand, self._freqs)
        return phaseCovOneLayer

    def _getZernikeCPSDOneTemporalFrequency(self, j, k, nLayer, t_freq):
        func, fPerp = self.integrandOfZernikeCPSD(j, k, nLayer, t_freq)
        return self._integrate(func, fPerp)

    def _getZernikeCPSDAllTemporalFrequenciesOneLayer(self, j, k, nLayer,
                                                      temp_freqs):
        return np.array([
            self._getZernikeCPSDOneTemporalFrequency(
                j, k, nLayer, t_freq) for t_freq in temp_freqs
        ])

    def _getZernikeCPSDAllLayers(self, j, k, temp_freqs):
        cpsd = np.array([
            self._getZernikeCPSDAllTemporalFrequenciesOneLayer(
                j, k, nLayer, temp_freqs) for nLayer
            in range(self._numberOfLayers)])
        cpsdTotal = cpsd.sum(axis=0)
        return cpsdTotal

    def _getGeneralZernikeCPSDOneTemporalFrequency(self, j, k, nLayer, t_freq):
        func, fPerp = self._integrandOfGeneralZernikeCPSD(j, k, nLayer, t_freq)
        return self._integrate(func, fPerp)

    def _getGeneralZernikeCPSDAllTemporalFrequenciesOneLayer(self, j, k,
                                                             nLayer,
                                                             temp_freqs):
        return np.array([
            self._getGeneralZernikeCPSDOneTemporalFrequency(
                j, k, nLayer, t_freq) for t_freq in temp_freqs
        ])

    def _getGeneralZernikeCPSDAllLayers(self, j, k, temp_freqs):
        cpsd = np.array([
            self._getGeneralZernikeCPSDAllTemporalFrequenciesOneLayer(
                j, k, nLayer, temp_freqs) for nLayer
            in range(self._numberOfLayers)])
        cpsdTotal = cpsd.sum(axis=0)
        return cpsdTotal

    def _getPhaseCPSDOneTemporalFrequency(self, nLayer, t_freq):
        func, fPerp = self.integrandOfPhaseCPSD(nLayer, t_freq)
        return self._integrate(func, fPerp)

    def _getPhaseCPSDAllTemporalFrequenciesOneLayer(self, nLayer,
                                                    temp_freqs):
        return np.array([
            self._getPhaseCPSDOneTemporalFrequency(nLayer, t_freq)
            for t_freq in temp_freqs
        ])

    def _getPhaseCPSDAllLayers(self, temp_freqs):
        phaseCPSD = np.array([
            self._getPhaseCPSDAllTemporalFrequenciesOneLayer(
                nLayer, temp_freqs)
            for nLayer in range(self._numberOfLayers)])
        phaseCPSDTotal = phaseCPSD.sum(axis=0)
        return phaseCPSDTotal

    def _integrandOfGeneralZernikeCPSD(self, j, k, nLayer, temp_freq):
        vl = self._windSpeed[nLayer]

        fPerp = self._freqs
        f = np.sqrt(fPerp ** 2 + (temp_freq / vl) ** 2)

        self._initializeGeometry(nLayer, f)
        self._initializeFunctionsForZernikeComputations(j, k, f)

        thWind = np.deg2rad(self._windDirection[nLayer])
        th0 = np.arccos(-temp_freq / (f * vl))
        th1 = th0 + thWind
        th2 = -th0 + thWind

        self._c4 = np.pi / 4 * (1 - self._deltaj) * ((-1) ** j - 1)
        self._c5 = np.pi / 4 * (1 - self._deltak) * ((-1) ** k - 1)

        integFunc = 2 * self._c0 * self._c1 * 1j**(0.5 * ((-1)**(
            self._nj + self._nk) - 1)) / (vl * np.pi) * \
            self._psd / f ** 2 * self._b1 * self._b2 * \
            (np.cos(2 * np.pi * f * self._sl * np.cos(th1 - self._thS) +
                    np.pi / 4 * ((-1)**(self._nj + self._nk) - 1)) *
             np.cos(self._mj * th1 + self._c4) *
             np.cos(self._mk * th1 + self._c5) +
             np.cos(2 * np.pi * f * self._sl * np.cos(th2 - self._thS) +
                    np.pi / 4 * ((-1)**(self._nj + self._nk) - 1)) *
             np.cos(self._mj * th2 + self._c4) *
             np.cos(self._mk * th2 + self._c5))
        return integFunc, fPerp

    def integrandOfZernikeCPSD(self, j, k, nLayer, temp_freq):
        vl = self._windSpeed[nLayer]

        fPerp = self._freqs
        f = np.sqrt(fPerp ** 2 + (temp_freq / vl) ** 2)

        self._initializeGeometry(nLayer, f)
        self._initializeFunctionsForZernikeComputations(j, k, f)

        thWind = np.deg2rad(self._windDirection[nLayer])
        th0 = np.arccos(-temp_freq / (f * vl))
        th1 = th0 + thWind
        th2 = -th0 + thWind

        self._c4 = np.pi / 4 * (1 - self._deltaj) * ((-1) ** j - 1)
        self._c5 = np.pi / 4 * (1 - self._deltak) * ((-1) ** k - 1)

        integFunc = self._c0 * self._c1 / (vl * np.pi) * \
            self._psd / f ** 2 * self._b1 * self._b2 * \
            (np.exp(-2 * 1j * np.pi * f * self._sl * np.cos(
                th1 - self._thS)) *
             np.cos(self._mj * th1 + self._c4) *
             np.cos(self._mk * th1 + self._c5) +
             np.exp(-2 * 1j * np.pi * f * self._sl * np.cos(
                 th2 - self._thS)) *
             np.cos(self._mj * th2 + self._c4) *
             np.cos(self._mk * th2 + self._c5))
        return integFunc, fPerp

    def integrandOfPhaseCPSD(self, nLayer, temp_freq):
        vl = self._windSpeed[nLayer]

        fPerp = self._freqs
        f = np.sqrt(fPerp ** 2 + (temp_freq / vl) ** 2)

        self._initializeGeometry(nLayer, f)
        self._initializeFunctionsForPhaseComputations(f)

        thWind = np.deg2rad(self._windDirection[nLayer])
        th0 = np.arccos(-temp_freq / (f * vl))
        th1 = th0 + thWind
        th2 = -th0 + thWind

        intFunc = 1. / vl * self._psd * (
            self._b1Phase - self._b2Phase * self._b3Phase) * (
                np.exp(-2 * 1j * np.pi * f * self._sl * np.cos(
                    th1 - self._thS)) +
            np.exp(-2 * 1j * np.pi * f * self._sl * np.cos(
                th2 - self._thS)))
        return intFunc, fPerp

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
        Return the covariance between the phase seen from source1 with
        aperture1 and the phase seen from source2 with aperture2.

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
        coefficients with index j and k describing the phase seen by aperture1
        and aperture2 observing, respectively, source1 and source2.
        The CPSD is a function of temporal frequency.

        Parameters
        ----------
        j: int
            Index of the Zernike coefficient (related to source1 on aperture1).
        k: int
            Index of the Zernike coefficient (related to source2 on aperture2).
        temp_freqs: numpy.ndarray
            Temporal frequencies array.

        Returns
        -------
        zernikeCPSD: `~astropy.units.quantity.Quantity`
            Zernike CPSD or matrix of Zernike CPSDs in [rad**2/Hz].
        """

        if (np.isscalar(j) and np.isscalar(k)):
            zernikeCPSD = self._getZernikeCPSDAllLayers(j, k, temp_freqs
                                                        ) * u.rad ** 2 / u.Hz

        elif (np.isscalar(j) and np.isscalar(k) is False):
            zernikeCPSD = np.array([
                self._getZernikeCPSDAllLayers(j, k_mode, temp_freqs)
                for k_mode in k]) * u.rad ** 2 / u.Hz

        elif (np.isscalar(j) is False and np.isscalar(k)):
            zernikeCPSD = np.array([
                self._getZernikeCPSDAllLayers(j_mode, k, temp_freqs)
                for j_mode in j]) * u.rad ** 2 / u.Hz

        else:
            zernikeCPSD = np.array([
                [self._getZernikeCPSDAllLayers(j_mode, k_mode, temp_freqs)
                    for k_mode in k]
                for j_mode in j]) * u.rad ** 2 / u.Hz

        return zernikeCPSD

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

        if (np.isscalar(j) and np.isscalar(k)):
            zernikeCPSD = self._getGeneralZernikeCPSDAllLayers(
                j, k, temp_freqs) * u.rad ** 2 / u.Hz

        elif (np.isscalar(j) and np.isscalar(k) is False):
            zernikeCPSD = np.array([
                self._getGeneralZernikeCPSDAllLayers(j, k_mode, temp_freqs)
                for k_mode in k]) * u.rad ** 2 / u.Hz

        elif (np.isscalar(j) is False and np.isscalar(k)):
            zernikeCPSD = np.array([
                self._getGeneralZernikeCPSDAllLayers(j_mode, k, temp_freqs)
                for j_mode in j]) * u.rad ** 2 / u.Hz

        else:
            zernikeCPSD = np.array([
                [self._getGeneralZernikeCPSDAllLayers(j_mode, k_mode,
                                                      temp_freqs)
                    for k_mode in k]
                for j_mode in j]) * u.rad ** 2 / u.Hz

        return zernikeCPSD

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

        phaseCPSD = self._getPhaseCPSDAllLayers(temp_freqs) * u.rad ** 2 / u.Hz
        return phaseCPSD

    def plotCPSD(self, cpsd, temp_freqs, func_part, scale, legend='',
                 wavelength=None):
        import matplotlib.pyplot as plt
        if wavelength is None:
            lam = self._cn2.DEFAULT_LAMBDA
        else:
            lam = wavelength
        m_to_nm = 1e18
        if func_part == 'real':
            if scale == 'log':
                plt.loglog(
                    temp_freqs,
                    np.abs(np.real(cpsd)) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='(real) ' + legend)
            elif scale == 'linear':
                plt.semilogx(
                    temp_freqs,
                    np.real(cpsd) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='(real) ' + legend)
        elif func_part == 'imag':
            if scale == 'log':
                plt.loglog(
                    temp_freqs,
                    np.abs(np.imag(cpsd)) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='(imag) ' + legend)
            elif scale == 'linear':
                plt.semilogx(
                    temp_freqs,
                    np.imag(cpsd) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='(imag) ' + legend)
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('CPSD [nm$^{2}$/Hz]')
