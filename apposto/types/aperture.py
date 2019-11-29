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
        Radius of the optical aperture in [m].

    cartes_coords: tuple of three floats
        Cartesian coordinates (x, y, z) of the aperture center
        in [m].
    '''

    def __init__(self, aperture_radius, cartes_coords):
        self._r = aperture_radius
        self._x = cartes_coords[0]
        self._y = cartes_coords[1]
        self._z = cartes_coords[2]

    def getCartesianCoords(self):
        return np.array([self._x, self._y, self._z])

    def getApertureRadius(self):
        return self._r
