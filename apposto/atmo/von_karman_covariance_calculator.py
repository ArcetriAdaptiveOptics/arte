'''
@author: giuliacarla
'''

import numpy as np
import astropy.units as u
from apposto.utils import von_karman_psd, math
import logging
from apposto.utils.zernike_generator import ZernikeGenerator


class VonKarmanSpatioTemporalCovariance():
    '''
    This class computes the spatio-temporal covariance between Zernike
    coefficients which represent the turbulence induced phase aberrations
    of two sources seen with two different circular apertures.


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
    cn2_profile: Cn2Profile object
        cn2 profile as obtained from the Cn2Profile class.
        (e.g. cn2_profile = apposto.atmo.cn2_profile.EsoEltProfiles.Q1())

    source1: GuideSource object
        Source geometry as obtained from GuideSource class.
        Here we consider rho as the angle (in arcsec) with respect to the
            z-axis.
        theta is always the angle (in degrees) with respect to the x-axis.
        (e.g. source1 = apposto.types.guide_source.GuideSource((1,90), 9e3)

    source2: GuideSource object
        Same as source1.

    aperture1: CircularOpticalAperture object
        Optical aperture geometry as obtained from CircularOpticalAperture
            class.
        (e.g. aperture1 = apposto.types.aperture.CircularOpticalAperture(
                                                    10, (0, 0, 0)))

    aperture2: CircularOpticalAperture object
        Same as aperture1.
    '''

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

    def setSource1(self, s1):
        self._source1 = s1

    def setSource2(self, s2):
        self._source2 = s2

    def setAperture1(self, ap1):
        self._ap1 = ap1

    def setAperture2(self, ap2):
        self._ap2 = ap2

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

    def layerProjectedAperturesSeparation(self, nLayer):
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

    def _initializeParams1(self, nLayer, spat_freqs):
        self._a1 = self._layerScalingFactor1(nLayer)
        self._a2 = self._layerScalingFactor2(nLayer)
        self._R1 = self._ap1Radius
        self._R2 = self._ap2Radius
        sep = self.layerProjectedAperturesSeparation(nLayer)
        self._sepMod = np.linalg.norm(sep)
        self._thS = np.arctan2(sep[1], sep[0])
        self._psd = self._VonKarmanPSDOneLayer(nLayer, spat_freqs)

    def _initializeParams2(self, j, k, spat_freqs):
        self._nj, self._mj = ZernikeGenerator.degree(j)
        self._nk, self._mk = ZernikeGenerator.degree(k)

        self._deltaj = math.kroneckerDelta(0, self._mj)
        self._deltak = math.kroneckerDelta(0, self._mk)

        self._b1 = np.array([math.besselFirstKind(
            self._nj + 1,
            2 * np.pi * f * self._R1 * (1 - self._a1)) for f in spat_freqs])
        self._b2 = np.array([math.besselFirstKind(
            self._nk + 1,
            2 * np.pi * f * self._R2 * (1 - self._a2)) for f in spat_freqs])

        self._c0 = (-1) ** self._mk * np.sqrt((
            self._nj + 1) * (self._nk + 1)) * np.complex(0, 1) ** (
            self._nj + self._nk) * 2 ** (
            1 - 0.5 * (self._deltaj + self._deltak))
        self._c1 = 1. / (
            np.pi * self._R1 * self._R2 * (1 - self._a1) * (1 - self._a2))

    def _integrate(self, int_func, int_var):
        i = np.complex(0, 1)
        return np.trapz(np.real(int_func), int_var) + \
            i * np.trapz(np.imag(int_func), int_var)

    def _computeZernikeCovarianceOneLayer(self, j, k, nLayer):
        self._initializeParams1(nLayer, self._freqs)
        self._initializeParams2(j, k, self._freqs)
        i = np.complex(0, 1)

        self._b3 = np.array([math.besselFirstKind(
            self._mj + self._mk,
            self._sepMod * 2 * np.pi * f) for f in self._freqs])
        self._b4 = np.array([math.besselFirstKind(
            np.abs(self._mj - self._mk),
            self._sepMod * 2 * np.pi * f) for f in self._freqs])
        self._c2 = np.pi / 4 * ((1 - self._deltaj) * ((-1) ** j - 1) +
                                (1 - self._deltak) * ((-1) ** k - 1))
        self._c3 = np.pi / 4 * ((1 - self._deltaj) * ((-1) ** j - 1) -
                                (1 - self._deltak) * ((-1) ** k - 1))

        integFunc = self._c0 * self._c1 * \
            self._psd / self._freqs * self._b1 * self._b2 * \
            (np.cos((self._mj + self._mk) * self._thS + self._c2) *
             i ** (3 * (self._mj + self._mk)) *
             self._b3 +
             np.cos((self._mj - self._mk) * self._thS + self._c3) *
             i ** (3 * np.abs(self._mj - self._mk)) *
             self._b4)
        return integFunc

    def _getZernikeCovarianceOneLayer(self, j, k, nLayer):
        func = self._computeZernikeCovarianceOneLayer(j, k, nLayer)
        return self._integrate(func, self._freqs)

    def _getZernikeCovarianceAllLayers(self, j, k):
        cov = np.array([
            self._getZernikeCovarianceOneLayer(j, k, nLayer) for nLayer
            in range(self._numberOfLayers)])
        return cov.sum()

    def getZernikeCovariance(self, j, k):
        '''
        Parameters
        ----------
        j: int (scalar or list, numpy array, tuple...)
            Index of the Zernike coefficient (related to source1 on
            aperture1).

        k: int (scalar or list, numpy array, tuple...)
            Index of the Zernike coefficient (related to source2 on
            aperture2).

        Returns
        -------
        cov: Zernike covariance or covariance matrix (n x m matrix if
            n and m are, respectively, the dimension of j and k).
        '''
        if (np.isscalar(j) and np.isscalar(k)):
            cov = self._getZernikeCovarianceAllLayers(j, k)

        elif (np.isscalar(j) and np.isscalar(k) is False):
            cov = np.array([
                self._getZernikeCovarianceAllLayers(j, k_mode)
                for k_mode in k])

        elif (np.isscalar(j) is False and np.isscalar(k)):
            cov = np.array([
                self._getZernikeCovarianceAllLayers(j_mode, k)
                for j_mode in j])

        else:
            cov = np.matrix([
                np.array([
                    self._getZernikeCovarianceAllLayers(j_mode, k_mode)
                    for k_mode in k])
                for j_mode in j])

        return cov * u.rad**2

    def integrandOfZernikeCPSD(self, j, k, nLayer, temp_freq):
        i = np.complex(0, 1)
        vl = self._windSpeed[nLayer]

        fPerp = self._freqs
        f = np.sqrt(fPerp ** 2 + (temp_freq / vl) ** 2)

        self._initializeParams1(nLayer, f)
        self._initializeParams2(j, k, f)

        thWind = np.deg2rad(self._windDirection[nLayer])
        th0 = np.array([np.arccos(-temp_freq / (sp_freq * vl))
                        for sp_freq in f])
        th1 = th0 + thWind
        th2 = -th0 + thWind

        self._c4 = np.pi / 4 * (1 - self._deltaj) * ((-1) ** j - 1)
        self._c5 = np.pi / 4 * (1 - self._deltak) * ((-1) ** k - 1)

        integFunc = self._c0 * self._c1 / (vl * np.pi) * \
            self._psd / f ** 2 * self._b1 * self._b2 * \
            (np.exp(-2 * i * np.pi * f * self._sepMod * np.cos(
                th1 - self._thS)) *
             np.cos(self._mj * th1 + self._c4) *
             np.cos(self._mk * th1 + self._c5) +
             np.exp(-2 * i * np.pi * f * self._sepMod * np.cos(
                 th2 - self._thS)) *
             np.cos(self._mj * th2 + self._c4) *
             np.cos(self._mk * th2 + self._c5))
        return integFunc, fPerp

    def _getZernikeCPSDOneTemporalFrequency(self, j, k, nLayer, t_freq):
        func, fPerp = self.integrandOfZernikeCPSD(j, k, nLayer, t_freq)
        return self._integrate(func, fPerp)

    def _getZernikeCPSDAllTemporalFrequenciesOneLayer(self, j, k, nLayer,
                                                      temp_freqs):
        return np.array([
            self._getZernikeCPSDOneTemporalFrequency(
                j, k, nLayer, t_freq) for t_freq in temp_freqs
        ])

    def getZernikeCPSD(self, j, k, temp_freqs):
        '''
        Return the Cross Power Spectral Density (CPSD) of the Zernike
        coefficients with index j and k describing the phase seen,
        respectively, on aperture1 from source1 and on aperture2
        from source2.
        The CPSD is a function of temporal frequency.

        Parameters
        ----------
        j: int
            Index of the Zernike coefficient (related to source1 on aperture1).
        k: int
            Index of the Zernike coefficient (related to source2 on aperture2).
        temp_freqs: numpy ndarray
            Temporal frequencies array.

        Returns
        -------
        cpsdTotal: numpy ndarray
            Total Zernike CPSD, that is the sum of all the layers' CPSD.
        '''
        cpsd = np.array([
            self._getZernikeCPSDAllTemporalFrequenciesOneLayer(
                j, k, nLayer, temp_freqs) for nLayer
            in range(self._numberOfLayers)])
        cpsdTotal = cpsd.sum(axis=0)
        return cpsdTotal * u.rad**2 / u.Hz

    def integrandOfPhaseCPSD(self, nLayer, temp_freq):
        i = np.complex(0, 1)
        vl = self._windSpeed[nLayer]

        fPerp = self._freqs
        f = np.sqrt(fPerp ** 2 + (temp_freq / vl) ** 2)

        self._initializeParams1(nLayer, f)

        thWind = np.deg2rad(self._windDirection[nLayer])
        th0 = np.array([np.arccos(-temp_freq / (sp_freq * vl))
                        for sp_freq in f])
        th1 = th0 + thWind
        th2 = -th0 + thWind

        k = (1 - self._a2) * self._R2 / ((1 - self._a1) * self._R1)

        arg1 = np.pi * self._R1 * (1 - self._a1)  # * (k - 1)
        arg2 = np.pi * self._R2 * (1 - self._a2)  # * (k - 1)
        b0 = np.array([math.besselFirstKind(1, 2 * arg1 * (k - 1) * freq)
                       for freq in f])
        b1 = np.array([math.besselFirstKind(1, 2 * arg1 * freq) for freq in f])
        b2 = np.array([math.besselFirstKind(1, 2 * arg2 * freq) for freq in f])

        intFunc = 1. / vl * self._psd * (
            b0 / (arg1 * (k - 1) * f) - b1 / (arg1 * f) * b2 / (arg2 * f)) * (
                np.exp(-2 * i * np.pi * f * self._sepMod * np.cos(
                    th1 - self._thS)) +
            np.exp(-2 * i * np.pi * f * self._sepMod * np.cos(
                th2 - self._thS)))
        return intFunc, fPerp

    def _getPhaseCPSDOneTemporalFrequency(self, nLayer, t_freq):
        func, fPerp = self.integrandOfPhaseCPSD(nLayer, t_freq)
        return self._integrate(func, fPerp)

    def _getPhaseCPSDAllTemporalFrequenciesOneLayer(self, nLayer,
                                                    temp_freqs):
        return np.array([
            self._getPhaseCPSDOneTemporalFrequency(nLayer, t_freq)
            for t_freq in temp_freqs
        ])

    def getPhaseCPSD(self, temp_freqs):
        phaseCPSD = np.array([
            self._getPhaseCPSDAllTemporalFrequenciesOneLayer(
                nLayer, temp_freqs)
            for nLayer in range(self._numberOfLayers)])
        return phaseCPSD.sum(axis=0) * u.rad**2 / u.Hz

    def plotCPSD(self, cpsd, temp_freqs, func_part, scale, wavelenght=None):
        import matplotlib.pyplot as plt
        if wavelenght is None:
            lam = self._cn2.DEFAULT_LAMBDA
        else:
            lam = wavelenght
        m_to_nm = 1e18
        if func_part == 'real':
            if scale == 'log':
                plt.loglog(
                    temp_freqs,
                    np.abs(np.real(cpsd)) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='Real')
            elif scale == 'linear':
                plt.semilogx(
                    temp_freqs,
                    np.real(cpsd) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='Real')
        elif func_part == 'imag':
            if scale == 'log':
                plt.loglog(
                    temp_freqs,
                    np.abs(np.imag(cpsd)) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='Imaginary')
            elif scale == 'linear':
                plt.semilogx(
                    temp_freqs,
                    np.imag(cpsd) *
                    (lam / (2 * np.pi)) ** 2 * m_to_nm,
                    '-', label='Imaginary')
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('CPSD [nm$^{2}$/Hz]')
