'''
Created on 3 mag 2022

@author: giuliacarla
'''

import numpy as np
from scipy.special import gamma
from mpmath import hyper, fp
from arte.utils.zernike_generator import ZernikeGenerator


class VonKarmanSpatialCovariance():
    '''
    This function computes covariance of Zernike components of wavefront
    perturbations due to Von Karman turbulence. It returns covariance 
    <a(j1)*a(j2)> where a(j1) and a(j2) are j1-th and j2-th Zernike components
    of wavefront perturbations due to Von Karman turbulence. Covariance is
    in (wavelength)**2*(D/r0)**(5/3) units, where D is the pupil diameter and
    r0 the Fried parameter. To have covariance of phase perurbations, use:
        
        4*pi**2*VON_COVAR(j1, j2) = <2*pi*a(j1) * 2*pi*a(j2)>.
    
        It is in (D/r0)**(5/3) units (rad**2)
    
    Parameters
    ----------
    j1: int
        Index of Zernike polynomial. j1 >= 1
    j2: int
        Index of Zernike polynomial. j2 >= 1
    L0: float
        Outerscale value in pupil diameter units.
        
    References
    ---------- 
    Winker D. M., 1991, JOSA, 8, 1569.
    Takato N. and Yamaguchi I., 1995, JOSA, 12, 958
    
    Examples
    -------- 
    Computes the average spatial variance of phase perturbation on a 
    circular pupil of diameter 1.5m due to the first 15 Zernike aberrations.
    Suppose r0 = 15cm, L0=40m.
    
    scale = (150./15)**(5./3)
    L0norm= (40./1.5)
    var = 0.
    for i in range(2,16):
        vk = VonKarmanSpatialCovariance(i, i, L0norm)
        var = var + vk.get_covariance()
        var = 4 * np.pi**2 * var * scale
    Strhel_Ratio = np.exp(-var) 
    '''
    
    def __init__(self, j1, j2, L0):
        self._j1 = j1
        self._j2 = j2
        self._L0 = L0
        self._n1, self._m1 = ZernikeGenerator.degree(j1)
        self._n2, self._m2 = ZernikeGenerator.degree(j2)
        
    def _compute_covariance(self):
        n1pn2 = (self._n1 + self._n2) / 2
        n1mn2 = (self._n1 - self._n2)/2
        x0 = np.pi / self._L0    
        
        
        result = (
            24./5.*gamma(6./5))**(5./6)*(gamma(11./6))**2/np.pi**(5./2)
        result = result*(-1)**(
            (self._n1 + self._n2 - 2 * self._m1) / 2
            ) * np.sqrt((self._n1 + 1) * (self._n2 + 1))
        result = result / np.sin(np.pi * (n1pn2 + 1./6))
        
        temp1  = np.sqrt(np.pi) / 2**(
            self._n1 + self._n2 + 3) * gamma(n1pn2 + 1) / gamma(n1pn2 + 1./6)
        temp1  = temp1 / gamma(11./6) / gamma(self._n1 + 2.) / gamma(
            self._n2 + 2.) * x0**(self._n1 + self._n2 - 5./3.)

        alpha  = [n1pn2 + 3./2, n1pn2 + 2, n1pn2 + 1]
        beta   = [n1pn2 + 1./6, self._n1 + 2, self._n2 + 2, 
                  self._n1 + self._n2 + 3]
        
        temp1  = temp1 * hyper(alpha, beta, x0**2)

        temp2  = - gamma(7./3) * gamma(17./6) / 2
        temp2  = temp2 / gamma(11./6 - n1pn2) / gamma(17./6 - n1mn2
                                                      )/gamma(17./6 + n1mn2)
        temp2  = temp2 / gamma(n1pn2 + 23./6) 

        alpha  = [11./6, 7./3, 17./6]
        beta   = [11./6 - n1pn2, 17./6 - n1mn2, 17./6 + n1mn2, 23./6 + n1pn2]
        
        temp2  = temp2 * hyper(alpha, beta, x0**2)
        return fp.mpf(result * (temp1 + temp2))
    
    def get_covariance(self):
        if self._m1!=self._m2 or (self._j1 + self._j2) %2==1 and self._m1!=0:
            return 0.
        else:
            return self._compute_covariance()