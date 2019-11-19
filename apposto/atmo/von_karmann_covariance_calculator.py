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
    Plantet et al. (2019) - "Spatio-temporal statistics of the turbulent
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
                                                    10, (1,90), 2)

    aperture2: type?
        optical aperture geometry as obtained from
            CircularOpticalAperture class
        (e.g. aperture2 = apposto.types.aperture.CircularOpticalAperture(
                                                    10, (1,90), 2)
    '''

    def __init__(self,
                 cn2_profile,
                 source1,
                 source2,
                 aperture1,
                 aperture2,
                 freqs,
                 logger=logging.getLogger('VK_COVARIANCE')):
        self._cn2 = cn2_profile
        self._source1 = source1
        self._source2 = source2
        self._ap1 = aperture1
        self._ap2 = aperture2
        self._freqs = freqs
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
        return self._source1.getSourceCartesianCoords()

    def source2Coords(self):
        return self._source2.getSourceCartesianCoords()

    def aperture1Radius(self):
        return self._ap1.getApertureRadius()

    def aperture2Radius(self):
        return self._ap2.getApertureRadius()

    def aperture1Coords(self):
        return self._ap1.getApertureCenterCartesianCoords()

    def aperture2Coords(self):
        return self._ap2.getApertureCenterCartesianCoords()

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

    def layerProjectedAperturesSeparation(self, nLayer):
        #         sCoords1 = self.source1Coords()[0:2]
        #         sCoords2 = self.source2Coords()[0:2]
        aCoord1 = self.aperture1Coords()  # [0:2]
        aCoord2 = self.aperture2Coords()  # [0:2]
        sCoord1 = self._source1.getSourcePolarCoords()
        sCoord2 = self._source2.getSourcePolarCoords()
        sCoord1New = np.array([np.sin(sCoord1[0]) * np.cos(sCoord1[1]),
                               np.sin(sCoord1[0]) * np.sin(sCoord1[1]),
                               np.cos(sCoord1[0])])
        sCoord2New = np.array([np.sin(sCoord2[0]) * np.cos(sCoord2[1]),
                               np.sin(sCoord2[0]) * np.sin(sCoord2[1]),
                               np.cos(sCoord2[0])])
#         sep = aCoords2 - aCoords1 + self.layerScalingFactor2(nLayer) * \
#             (sCoords2 - aCoords2) - self.layerScalingFactor1(nLayer) * \
#             (sCoords1 - aCoords1)
        sep = aCoord2 - aCoord1 + (self._layersAlt[nLayer] - aCoord2[2]) * \
            sCoord2New / sCoord2New[2] - \
            (self._layersAlt[nLayer] - aCoord1[2]) * \
            sCoord1New / sCoord1New[2]
        return sep

    def _VonKarmannPsdOneLayers(self, nLayer, freqs):
        r0 = self._cn2.r0s()[nLayer]
        L0 = self._cn2.outer_scale()
        vk = von_karmann_psd.VonKarmannPsd(r0, L0)
        psd = vk.spatial_psd(freqs)
        return psd

    def _zernikeCovarianceOneLayer(self, j, k, nLayer):
        self._logger.debug('\n LAYER #%d' % (nLayer))

        i = np.complex(0, 1)
        self._logger.debug('Computing a1')
        a1 = self.layerScalingFactor1(nLayer)
        self._logger.debug('Computing a2')
        a2 = self.layerScalingFactor2(nLayer)
        self._logger.debug('Getting R1')
        R1 = self.aperture1Radius()
        self._logger.debug('Getting R2')
        R2 = self.aperture2Radius()
        self._logger.debug('Computing s')
        sep = self.layerProjectedAperturesSeparation(nLayer)
        s = np.linalg.norm(sep)

        self._logger.debug('Computing theta_s')
        self._thS = np.arctan2(sep[1], sep[0])
        self._logger.debug('Getting VonKarmann PSD')
        self._psd = self._VonKarmannPsdOneLayers(nLayer, self._freqs)

        dummyNumb = 2
        zern = zernike_generator.ZernikeGenerator(dummyNumb)
        self._logger.debug('Computing zernike orders')
        self._nj, self._mj = zern.degree(j)
        self._nk, self._mk = zern.degree(k)

        self._logger.debug('Getting deltas')
        self._deltaj = math.kroneckerDelta(0, self._mj)
        self._deltak = math.kroneckerDelta(0, self._mk)

        self._logger.debug('Computing bessel1')
        self._b1 = np.array([math.besselFirstKind(
            self._nj + 1,
            2 * np.pi * f * R1 * (1 - a1)) for f in self._freqs])
        self._logger.debug('Computing bessel2')
        self._b2 = np.array([math.besselFirstKind(
            self._nk + 1,
            2 * np.pi * f * R2 * (1 - a2)) for f in self._freqs])
        self._logger.debug('Computing bessel3')
        self._b3 = np.array([math.besselFirstKind(
            self._mj + self._mk,
            s * 2 * np.pi * f) for f in self._freqs])
        self._logger.debug('Computing bessel4')
        self._b4 = np.array([math.besselFirstKind(
            np.abs(self._mj - self._mk),
            s * 2 * np.pi * f) for f in self._freqs])

        self._logger.debug('Computing c1')
        self._c1 = (-1)**self._mk * np.sqrt((
            self._nj + 1) * (self._nk + 1)) * np.complex(0, 1)**(
            self._nj + self._nk) * 2**(
            1 - 0.5 * (self._deltaj + self._deltak))
        self._logger.debug('Computing c2')
        self._c2 = np.pi / 4 * ((1 - self._deltaj) * ((-1)**j - 1) +
                                (1 - self._deltak) * ((-1)**k - 1))
        self._c3 = np.pi / 4 * ((1 - self._deltaj) * ((-1)**j - 1) -
                                (1 - self._deltak) * ((-1)**k - 1))

        self._logger.debug('Computing covariance')
        self._integFunc = self._c1 * 1 / (
            np.pi * R1 * R2 * (1 - a1) * (1 - a2)) * \
            self._psd / self._freqs * self._b1 * self._b2 * \
            (np.cos((self._mj + self._mk) * self._thS + self._c2) *
             i**(3 * (self._mj + self._mk)) *
             self._b3 +
             np.cos((self._mj - self._mk) * self._thS + self._c3) *
             i**(3 * np.abs(self._mj - self._mk)) *
             self._b4)

        self._covOneLayer = np.trapz(np.real(self._integFunc), self._freqs) + \
            np.trapz(np.imag(self._integFunc), self._freqs)

    def _getZernikeCovarianceOneLayer(self, j, k, nLayer):
        self._zernikeCovarianceOneLayer(j, k, nLayer)
        return self._covOneLayer

    def getZernikeCovariance(self, j, k):
        self._covAllLayers = np.array([
            self._getZernikeCovarianceOneLayer(j, k, nLayer) for nLayer
            in range(self._layersAlt.shape[0])])
        return self._covAllLayers.sum()

    def getZernikeCovarianceMatrix(self, j_vector, k_vector):
        matr = np.matrix([
            np.array([self.getZernikeCovariance(j_vector[j], k_vector[i])
                      for i in range(k_vector.shape[0])])
            for j in range(j_vector.shape[0])])
        return matr
