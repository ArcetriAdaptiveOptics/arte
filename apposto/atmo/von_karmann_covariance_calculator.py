'''
@author: giuliacarla
'''


import numpy as np
from apposto.utils import von_karmann_psd, math, zernike_generator
#import math


class VonKarmannSpatioTemporalCovariance():
    '''
    This class computes the spatio-temporal covariance between Zernike
    coefficients which represent the turbulence induced phase aberrations
    of two sources seen with two different circular apertures.


    References
    ----------
    Plantét et al. (2019) - "Spatio-temporal statistics of the turbulent
        Zernike coefficients and piston-removed phase from two distinct beams".

    Plantét et al. (2018) - "LO WFS of MAORY: performance and sky coverage
        assessment."

    Whiteley et al. (1998) - "Temporal properties of the Zernike expansion
        coefficients of turbulence-induced phase aberrations for aperture
        and source motion".




    Parameters
    ----------
    cn2_profile: cn2 profile as obtained from the Cn2Profile class
            (e.g. cn2_profile = apposto.atmo.cn2_profile.EsoEltProfiles.Q1())

    source1:

    source2:

    aperture1:

    aperture2:
    '''

    def __init__(self, cn2_profile, source1, source2, aperture1, aperture2):
        self._cn2 = cn2_profile
        self._source1 = source1
        self._source2 = source2
        self._ap1 = aperture1
        self._ap2 = aperture2
        self._layersAlt = cn2_profile.layersDistance()
        self._windSpeed = cn2_profile.windSpeed()
        self._windDirection = cn2_profile.windDirection()

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

    def layerScalingFactor2(self, nLayer):
        a2 = self._layerScalingFactor(
            self._layersAlt[nLayer],
            self.source2Coords()[2],
            self.aperture2Coords()[2])
        return a2

    def _layerScalingFactor(self, zLayer, zSource, zAperture):
        scalFact = (zLayer - zAperture) / (zSource - zAperture)
        return scalFact

    def layerProjectedAperturesSeparation(self, nLayer):
        sep = self.aperture2Coords() - self.aperture1Coords() + \
            self.layerScalingFactor2(nLayer) * \
            (self.source2Coords() - self.aperture2Coords()) - \
            self.layerScalingFactor1(nLayer) * \
            (self.source1Coords() - self.aperture1Coords())
        return sep

    def _VonKarmannPsdOfAllLayers(self, freqs):
        vk = von_karmann_psd.VonKarmannPsd(self._cn2)
        psd = vk.getVonKarmannPsdOfAllLayers(freqs)
        return psd

    def _zernikeCovarianceMatrixOneLayer(self, j, k, nLayer, freqs):
        print('Computing a1')
        a1 = self.layerScalingFactor1(nLayer)
        print('Computing a2')
        a2 = self.layerScalingFactor2(nLayer)
        print('Getting R1')
        R1 = self.aperture1Radius()
        print('Getting R2')
        R2 = self.aperture2Radius()
        print('Computing s')
        s = np.linalg.norm(self.layerProjectedAperturesSeparation(nLayer))

        dummyNumb = 2
        zern = zernike_generator.ZernikeGenerator(dummyNumb)
        print('Computing zernike orders')
        self._nj, self._mj = zern._degree(j)
        self._nk, self._mk = zern._degree(k)

        print('Getting deltas')
        self._deltaj = math.kroneckerDelta(0, self._mj)
        self._deltak = math.kroneckerDelta(0, self._mk)

        print('Computing bessel1')
        self._b1 = np.array([math.besselFirstKind(
            self._nj + 1,
            2 * np.pi * f * R1 * (1 - a1)) for f in freqs])
        print('Computing bessel2')
        self._b2 = np.array([math.besselFirstKind(
            self._nk + 1,
            2 * np.pi * f * R2 * (1 - a2)) for f in freqs])
        print('Computing bessel3')
        self._b3 = np.array([math.besselFirstKind(
            self._mj + self._mk,
            s * 2 * np.pi * f) for f in freqs])
        print('Computing bessel4')
        self._b4 = np.array([math.besselFirstKind(
            np.abs(self._mj - self._mk),
            s * 2 * np.pi * f) for f in freqs])

        print('Computing c1')
        self._c1 = (-1)**self._mk * np.sqrt((
            self._nj + 1) * (self._nk + 1)) * np.complex(0, 1)**(
            self._nj + self._nk) * 2**(
            1 - 0.5 * (self._deltaj + self._deltak))
        print('Computing c2')
        self._c2 = np.pi / 4 * ((1 - self._deltaj) * ((-1)**j - 1) -
                                (1 - self._deltak) * ((-1)**k - 1))
        pass


#     def _costant(self):
#         c = (-1)**self._mk * np.sqrt((
#             self._nj+1)*(self._nk+1)) * np.complex(0,1)**(
#             self._nj + self._nk) * 2**(
#             1-0.5*(self.KroneckerDelta(self._mj, 0) +
#                    self.KroneckerDelta(self._mk, 0)))
#         return c
#
#
#     def _integrand(self, i):
#         def bess(order, arg):
#             return special.jv(order, arg)
#
#         def kDelta(m, n):
#             if m==n:
#                 delta = 1
#             else:
#                 delta = 0
#             return delta
#
#         h = self._hs[i]
#         d = np.abs(self.starsDistance())
#         psdFreqFunc = self._VonKarmannPSDasFrequencyFunction(self._r0s[i])
#         return lambda f : 1/(np.pi * self._R**2 * (1 - h/self._z1) *
#             (1 - h/self._z2)) * (psdFreqFunc(f)/f) * bess(self._nj+1,
#             2*np.pi*f*self._R*(1-h/self._z1)) * bess(self._nk+1,
#             2*np.pi*f+self._R*(1-h/self._z2)) * (
#             np.cos((self._mj+self._mk) * d + np.pi/4 *
#             ((1 - kDelta(0, self._mj)) * ((-1)**self._j - 1) +
#             (1 - kDelta(0, self._mk)) * ((-1)**self._k - 1))) *
#             np.complex(0,1)**(3*(self._mj+self._mk)) *
#             bess(self._mj+self._mk, (2*np.pi*f*h*d)) +
#             np.cos((self._mj-self._mk) * d + np.pi/4 *
#             ((1 - kDelta(0, self._mj)) * ((-1)**self._j - 1) -
#             (1 - kDelta(0, self._mk)) * ((-1)**self._k - 1))) *
#             np.complex(0,1)**(3*np.abs(self._mj-self._mk)) *
#             bess(np.abs(self._mj-self._mk), (2*np.pi*f*h*d)))
#
#     def _computeFrequencyIntegral(self, func):
#         def realFunc(f):
#             return np.real(func(f))
#         def imagFunc(f):
#             return np.imag(func(f))
#         realInt = integrate.quad(realFunc, 1e-3, 1e3)
#         imagInt = integrate.quad(imagFunc, 1e-3, 1e3)
#         return realInt, imagInt, np.complex(realInt[0], imagInt[0])
#
#     def getFrequencyIntegral(self):
#         return np.array([
#             self._computeFrequencyIntegral(self._integrand(i))[2] for i
#             in range(self._hs.shape[0])])
