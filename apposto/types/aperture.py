'''
@author: giuliacarla
'''


class CircularOpticalAperture():
    '''
    This class defines the geometry of a circular optical aperture.
    
    Parameters
    ----------
    
    '''
    
    
    def __init__(self, aperture_radius, polar_coords, height):
        self._r = aperture_radius
        self._rho = polar_coords[0]
        self._theta = polar_coords[1]
        self._h = height