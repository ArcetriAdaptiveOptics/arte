'''
@author: giuliacarla
'''


import numpy as np


class CircularOpticalAperture():
    '''
    This class defines the geometry of a circular optical aperture.
    
    Parameters
    ----------
    aperture_radius: float
        Radius of the optical aperture in [meters].
    
    polar_coords: tuple of two floats
        Polar coordinates (rho, theta) of the aperture center
        in [arcsec, degrees].
    
    height: float
        Height of the aperture center in [meters].    
    '''
    
    
    def __init__(self, aperture_radius, polar_coords, height):
        self._r = aperture_radius
        self._rho = polar_coords[0]
        self._theta = polar_coords[1]
        self._z = height
        
    @staticmethod    
    def fromDegToRad(angle_deg):
        return angle_deg*np.pi/180
        
    @staticmethod    
    def fromPolarToCartesian(rho, thetaRad):
        x = rho*np.cos(thetaRad)
        y = rho*np.sin(thetaRad)
        return x, y
    
    def getApertureRadius(self):
        return self._r
    
    def getApertureCenterCartesianCoords(self):
        x, y = self.fromPolarToCartesian(self._rho,
                                         self.fromDegToRad(self._theta))
        return np.array([x, y, self._z])
    
    def getApertureCenterPolarCoords(self):
        return np.array([self._rho, self._theta, self._z])