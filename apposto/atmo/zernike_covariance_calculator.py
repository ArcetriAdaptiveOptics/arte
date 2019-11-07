'''
@author: giuliacarla
'''


import numpy as np
from scipy import special
from apposto.utils import von_karmann_psd
#from apposto.atmo.cn2_profile import EsoEltProfiles


class ZernikeSpatioTemporalCovariance():
    '''
    This class computes the spatio-temporal covariance between Zernike 
    coefficients which represent the turbulence induced phase aberrations
    of two sources seen with two different circular apertures. 
    
    
    References
    ----------
    Plantét et al. (2019) - "Spatio-temporal statistics of the turbulent
        Zernike coefficients and piston-removed phase from two distinct beams".
        
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
        
    def _layerScalingFactor(self, z_layer, z_source, z_aperture):
        return (z_layer - z_aperture)/(z_source - z_aperture)
            
    def _layerProjectedAperturesSeparation(self, z_layer):
#         a1_l = self._layerScalingFactor(z_layer, z_source1, z_aperture1)
#         a2_l = self._layerScalingFactor(z_layer, z_source2, z_aperture2)
        pass
    
    def _VonKarmannPsdOfAllLayers(self, freqs):
        vk = von_karmann_psd.VonKarmannPsd(self._cn2)
        psd = vk.getVonKarmannPsdOfAllLayers(freqs)
        return psd
    
    @staticmethod
    def getOrdersFromZernikeIndex(zern_idx):
        n = int(0.5 * (np.sqrt(8 * zern_idx - 7) - 3)) + 1
        cn = n * (n + 1) / 2 + 1
        if n % 2 == 0:
            m = int(zern_idx - cn + 1) // 2 * 2
        else:
            m = int(zern_idx - cn) // 2 * 2 + 1
        radialOrder = n
        azimuthalOrder = m
        return radialOrder, azimuthalOrder
    
    def covarianceMatrix(self, j, k):
        self._nj, self._mj = self.getOrdersFromZernikeIndex(j)
        self._nk, self._mk = self.getOrdersFromZernikeIndex(k)
        
        cost = (-1)**self._mk * np.sqrt((
            self._nj+1)*(self._nk+1)) * np.complex(0,1)**(
            self._nj + self._nk) * 2**(
            1-0.5*(self.delta(self._mj, 0) + 
                   self.delta(self._mk, 0))) #* (
            #self._windSpeed*np.pi**2)
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