'''
@author: giuliacarla
'''


import numpy as np
from apposto.utils import von_karmann_psd, math, zernike_generator
import logging


class VonKarmannSpatioTemporalCovariance():
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
    cn2_profile: type?
        cn2 profile as obtained from the Cn2Profile class
        (e.g. cn2_profile = apposto.atmo.cn2_profile.EsoEltProfiles.Q1())

    source1: type?
        source geometry as obtained from GuideSource class
        (e.g. source1 = apposto.types.guide_source.GuideSource((1,90), 9e3)

    source2: type?
        source geometry as obtained from GuideSource class
        (e.g. source2 = apposto.types.guide_source.GuideSource((1,90), 9e3)

    aperture1: type?
        optical aperture geometry as obtained from
            CircularOpticalAperture class
        (e.g. aperture1 = apposto.types.aperture.CircularOpticalAperture(
                                                    10, (0, 0, 0)))

    aperture2: type?
        optical aperture geometry as obtained from
            CircularOpticalAperture class
        (e.g. aperture2 = apposto.types.aperture.CircularOpticalAperture(
                                                    10, (5, 5, 5)))
    '''

    def __init__(self,
                 cn2_profile,
                 source1,
                 source2,
                 aperture1,
                 aperture2,
                 spat_freqs,
                 logger=logging.getLogger('VK_COVARIANCE')):
        self._cn2 = cn2_profile
        self._source1 = source1
        self._source2 = source2
        self._ap1 = aperture1
        self._ap2 = aperture2
        self._freqs = spat_freqs
        self._logger = logger
        self._layersAlt = cn2_profile.layers_distance()
        self._windSpeed = cn2_profile.wind_speed()
        self._windDirection = cn2_profile.wind_direction()

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

    def source1Coords(self):
        return self._source1.getSourcePolarCoords()

    def source2Coords(self):
        return self._source2.getSourcePolarCoords()

    def aperture1Radius(self):
        return self._ap1.getApertureRadius()

    def aperture2Radius(self):
        return self._ap2.getApertureRadius()

    def aperture1Coords(self):
        #         polar1 = self._ap1.getApertureCenterPolarCoords()
        #         cartes1 = np.array([
        #             polar1[2] * np.cos(np.deg2rad(polar1[1])) /
        #             np.cos(np.deg2rad(polar1[0])),
        #             polar1[2] * np.sin(np.deg2rad(polar1[1])) /
        #             np.cos(np.deg2rad(polar1[0])),
        #             polar1[2]])
        #         return cartes1
        return self._ap1.getCartesianCoords()

    def aperture2Coords(self):
        #         polar2 = self._ap2.getApertureCenterPolarCoords()
        #         cartes2 = np.array([
        #             polar2[2] * np.cos(np.deg2rad(polar2[1])) /
        #             np.cos(np.deg2rad(polar2[0])),
        #             polar2[2] * np.sin(np.deg2rad(polar2[1])) /
        #             np.cos(np.deg2rad(polar2[0])),
        #             polar2[2]])
        #         return cartes2
        return self._ap2.getCartesianCoords()

    def layerScalingFactor1(self, nLayer):
        a1 = self._layerScalingFactor(
            self._layersAlt[nLayer],
            self.source1Coords()[2],
            self.aperture1Coords()[2])
        return a1

#     def layerScalingFactor1(self, nLayer):
#         a1 = self._layerScalingFactor(self.source1Coords(),
#                                       self.aperture1Coords(),
#                                       nLayer)
#         return a1

    def layerScalingFactor2(self, nLayer):
        a2 = self._layerScalingFactor(
            self._layersAlt[nLayer],
            self.source2Coords()[2],
            self.aperture2Coords()[2])
        return a2

#     def layerScalingFactor2(self, nLayer):
#         a2 = self._layerScalingFactor(self.source2Coords(),
#                                       self.aperture2Coords(),
#                                       nLayer)
#        return a2

    def _layerScalingFactor(self, zLayer, zSource, zAperture):
        scalFact = (zLayer - zAperture) / (zSource - zAperture)
        return scalFact

#     def _layerScalingFactor(self, source_coord, aperture_coord, n_layer):
#         dist_versor = (source_coord - aperture_coord) / np.linalg.norm(
#             source_coord - aperture_coord)
#         scalFact = (self._layersAlt[n_layer] - aperture_coord[2]) * \
#             dist_versor / dist_versor[2]
#         return scalFact

    def _getCleverVersorCoords(self, s_coord):
        return np.array([
            np.sin(np.deg2rad(s_coord[0] / 3600)) *
            np.cos(np.deg2rad(s_coord[1])),
            np.sin(np.deg2rad(s_coord[0] / 3600)) *
            np.sin(np.deg2rad(s_coord[1])),
            np.cos(np.deg2rad(s_coord[0] / 3600))])

    def cleverVersor1Coords(self):
        return self._getCleverVersorCoords(self.source1Coords())

    def cleverVersor2Coords(self):
        return self._getCleverVersorCoords(self.source2Coords())

    def layerProjectedAperturesSeparation(self, nLayer):
        #         sCoords1 = self.source1Coords()[0:2]
        #         sCoords2 = self.source2Coords()[0:2]
        aCoord1 = self.aperture1Coords()  # [0:2]
        aCoord2 = self.aperture2Coords()  # [0:2]
#         sep = aCoords2 - aCoords1 + self.layerScalingFactor2(nLayer) * \
#             (sCoords2 - aCoords2) - self.layerScalingFactor1(nLayer) * \
#             (sCoords1 - aCoords1)
        vCoord1 = self.cleverVersor1Coords()
        vCoord2 = self.cleverVersor2Coords()
        sep = aCoord2 - aCoord1 + (self._layersAlt[nLayer] - aCoord2[2]) * \
            vCoord2 / vCoord2[2] - \
            (self._layersAlt[nLayer] - aCoord1[2]) * \
            vCoord1 / vCoord1[2]
        return sep

    def _VonKarmannPsdOneLayers(self, nLayer, freqs):
        r0 = self._cn2.r0s()[nLayer]
        L0 = self._cn2.outer_scale()
        vk = von_karmann_psd.VonKarmannPsd(r0, L0)
        psd = vk.spatial_psd(freqs)
        return psd

    def _initializeParams(self, j, k, nLayer):
        self._a1 = self.layerScalingFactor1(nLayer)
        self._a2 = self.layerScalingFactor2(nLayer)
        self._R1 = self.aperture1Radius()
        self._R2 = self.aperture2Radius()
        self._sep = self.layerProjectedAperturesSeparation(nLayer)
        self._sepMod = np.linalg.norm(self._sep)
        self._thS = np.arctan2(self._sep[1], self._sep[0])

        self._psd = self._VonKarmannPsdOneLayers(nLayer, self._freqs)

        dummyNumb = 2
        zern = zernike_generator.ZernikeGenerator(dummyNumb)
        self._nj, self._mj = zern.degree(j)
        self._nk, self._mk = zern.degree(k)

        self._deltaj = math.kroneckerDelta(0, self._mj)
        self._deltak = math.kroneckerDelta(0, self._mk)

        self._b1 = np.array([math.besselFirstKind(
            self._nj + 1,
            2 * np.pi * f * self._R1 * (1 - self._a1)) for f in self._freqs])
        self._b2 = np.array([math.besselFirstKind(
            self._nk + 1,
            2 * np.pi * f * self._R2 * (1 - self._a2)) for f in self._freqs])

        self._c0 = (-1)**self._mk * np.sqrt((
            self._nj + 1) * (self._nk + 1)) * np.complex(0, 1)**(
            self._nj + self._nk) * 2**(
            1 - 0.5 * (self._deltaj + self._deltak))
        self._c1 = 1. / (
            np.pi * self._R1 * self._R2 * (1 - self._a1) * (1 - self._a2))

    def _zernikeCovarianceOneLayer(self, j, k, nLayer):
        self._initializeParams(j, k, nLayer)
        i = np.complex(0, 1)

        self._b3 = np.array([math.besselFirstKind(
            self._mj + self._mk,
            self._sepMod * 2 * np.pi * f) for f in self._freqs])
        self._b4 = np.array([math.besselFirstKind(
            np.abs(self._mj - self._mk),
            self._sepMod * 2 * np.pi * f) for f in self._freqs])
        self._c2 = np.pi / 4 * ((1 - self._deltaj) * ((-1)**j - 1) +
                                (1 - self._deltak) * ((-1)**k - 1))
        self._c3 = np.pi / 4 * ((1 - self._deltaj) * ((-1)**j - 1) -
                                (1 - self._deltak) * ((-1)**k - 1))

        self._integCovFunc = self._c0 * self._c1 * \
            self._psd / self._freqs * self._b1 * self._b2 * \
            (np.cos((self._mj + self._mk) * self._thS + self._c2) *
             i**(3 * (self._mj + self._mk)) *
             self._b3 +
             np.cos((self._mj - self._mk) * self._thS + self._c3) *
             i**(3 * np.abs(self._mj - self._mk)) *
             self._b4)

        self._covOneLayer = np.trapz(np.real(self._integCovFunc),
                                     self._freqs)  # + \
        #np.trapz(np.imag(self._integCovFunc), self._freqs)

    def _getZernikeCovarianceOneLayer(self, j, k, nLayer):
        self._zernikeCovarianceOneLayer(j, k, nLayer)
        return self._covOneLayer

#     def getZernikeCovarianceMatrixOneLayer(self, j_vector, k_vector, nLayer):
#         matr = np.matrix([
#             np.array([
#                 self.getZernikeCovarianceOneLayer(
#                     j_vector[j], k_vector[i], nLayer)
#                 for i in range(k_vector.shape[0])])
#             for j in range(j_vector.shape[0])])
#         return matr

#     def getZernikeCovariance(self, j, k):
#         self._covAllLayers = np.array([
#             self.getZernikeCovarianceOneLayer(j, k, nLayer) for nLayer
#             in range(self._layersAlt.shape[0])])
#         return self._covAllLayers.sum()

    def getZernikeCovariance(self, j, k, nLayer=None):
        if nLayer is None:
            cov = np.array([
                self._getZernikeCovarianceOneLayer(j, k, nLayer) for nLayer
                in range(self._layersAlt.shape[0])])
            return cov.sum()
        else:
            cov = self._getZernikeCovarianceOneLayer(j, k, nLayer)
            return cov

    def getZernikeCovarianceMatrix(self, j_vector, k_vector, nLayer=None):
        matr = np.matrix([
            np.array([
                self.getZernikeCovariance(j_vector[j], k_vector[i], nLayer)
                for i in range(k_vector.shape[0])])
            for j in range(j_vector.shape[0])])
        return matr

    def _zernikeCPSD(self, j, k, nLayer, temp_freq):
        vl = self._windSpeed[nLayer]
        fPerp = self._freqs
        f = np.sqrt(fPerp**2 + (temp_freq / vl)**2)
        self.setSpatialFrequencies(f)

        self._initializeParams(j, k, nLayer)

        thWind = np.deg2rad(self._windDirection[nLayer])
        self._th0 = np.array([np.arccos(- temp_freq / (sp_freq * vl))
                              for sp_freq in f])
        self._th1 = self._th0 + thWind
        self._th2 = - self._th0 + thWind

        self._c4 = np.pi / 4 * (1 - self._deltaj) * ((-1)**j - 1)
        self._c5 = np.pi / 4 * (1 - self._deltak) * ((-1)**k - 1)

        self._integCPSDFunc = 2 * self._c0 * self._c1 / (vl * np.pi) * \
            self._psd / self._freqs**2 * self._b1 * self._b2 * \
            (np.cos(2 * np.pi * self._freqs *
                    self._sepMod * np.cos(self._th1 - self._thS)) *
             np.cos(self._mj * self._th1 + self._c4) *
             np.cos(self._mk * self._th1 + self._c5) +
             np.cos(2 * np.pi * self._freqs *
                    self._sepMod * np.cos(self._th2 - self._thS)) *
             np.cos(self._mj * self._th2 + self._c4) *
             np.cos(self._mk * self._th2 + self._c5))

        self._cpsd = np.trapz(np.real(self._integCPSDFunc),
                              self._freqs)  # + \
#            np.trapz(np.imag(self._integCPSDFunc), self._freqs)

    def _getZernikeCPSDOneTempFreq(self, j, k, nLayer, t_freq):
        self._zernikeCPSD(j, k, nLayer, t_freq)
        return self._cpsd

    def getZernikeCPSD(self, j, k, nLayer, temp_freqs):
        return np.array([
            self._getZernikeCPSDOneTempFreq(j, k, nLayer, t_freq) for t_freq
            in temp_freqs])
