
from arte.utils.help import add_help
import astropy.units as u
import numpy as np

@add_help(classmethod=True)
class Telescope():

    @classmethod
    def diameter(cls):
        ''' Returns the telescope diameter'''
        return cls._diameter

    @classmethod
    def obstruction(cls):
        '''Returns the telescope obstruction, in percentage [0..1]'''
        return cls._obstruction

    @classmethod
    def area(cls):
        '''Returns the effective telescope collecting area'''
        r = cls._diameter/2
        obs = cls._obstruction*r 
        return np.pi*(r**2) - np.pi*(obs**2)


class VLT(Telescope):
    ''' VLT data '''
    _diameter = 8.120 * u.m
    _obstruction = 0.1592


class LBT(Telescope):
    ''' LBT data '''
    _diameter = 8.222 * u.m
    _obstruction = 0.111


class Magellan(Telescope):
    ''' Magellan telescope data '''

    _diameter = 6.5 * u.m
    _obstruction = 0.29


