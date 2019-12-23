'''
@author: giuliacarla
'''


import numpy as np
import astropy.units as u
from astropy.units.quantity import Quantity


class GuideSource():
    '''
    This class defines the geometry of the guide source of interest.

    Parameters
    ----------
    polar_coords: tuple of two floats
        Source polar coordinates (rho, theta) in [arcsec, degrees].
        theta is positive if generated by a counterclockwise rotation.

    height: float
        Source height in [meters].
    '''

    def __init__(self, polar_coords, height):
        self._rho = polar_coords[0]
        self._theta = polar_coords[1]
        self._z = height

    @staticmethod
    def _quantitiesToValue(quantity):
        if isinstance(quantity, Quantity):
            value = quantity.value
        else:
            value = quantity
        return value

    @staticmethod
    def fromPolarToCartesian(rho, theta, z):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y, z

    def getSourceCartesianCoords(self):
        x, y, z = self.fromPolarToCartesian(self._quantitiesToValue(self._rho),
                                            np.deg2rad(
            self._quantitiesToValue(
                self._theta)),
            self._quantitiesToValue(self._z))
        return [x * u.arcsec, y * u.arcsec, z * u.m]

    def getSourcePolarCoords(self):
        if (isinstance(self._rho, Quantity) and (self._theta, Quantity)
                and (self._z, Quantity)):
            return [self._rho, self._theta, self._z]
        else:
            return [self._rho * u.arcsec, self._theta * u.deg, self._z * u.m]
