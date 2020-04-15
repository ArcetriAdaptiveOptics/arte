'''
@author: lbusoni
'''
import numpy as np
from astropy.units.quantity import Quantity
from arte.utils.constants import Constants
from arte.utils.package_data import dataRootDir
import os
from astropy.io import fits
import astropy.units as u
from collections.abc import Iterable

"""

k = 2pi/lambda
J[i] = dh[i] * Cn2[i]
r0[i] = (0.422727 * airmass * k**2 J[i]) ** (-3/5)

r0 = (Sum r0[i]**(-5/3) ) ** (-3/5) = (Sum 0.423 airmass k**2 J[i]) ** (-3/5)
theta0 = (2.914 k**2 airmass**(8/3) Sum(J*h**(5/3))) ** (-3/5)
"""


class Cn2Profile(object):

    DEFAULT_LAMBDA = 500e-9
    DEFAULT_AIRMASS = 1.0

    def __init__(self,
                 layersJs,
                 layersL0,
                 layersAltitude,
                 layersWindSpeed,
                 layersWindDirection):
        """
        Cn2 profile constructor

        Parameters:
            layersJs (ndarray): array of layers Js in meters**(1/3)
            layersL0 (ndarray): array of layers outer-scale L0 in meters
            layersAltitude (ndarry): array of layers altitude at Zenith in
                meters

        Every array must be 1D and have the same size, defining the number of
            layers of the profile
        All parameters defined at zenith
        """
        self._layersJs = np.array(layersJs)
        self._layersL0 = np.array(layersL0)
        self._layersAltitudeInMeterAtZenith = np.array(layersAltitude)
        self._layersWindSpeed = np.array(layersWindSpeed)
        self._layersWindDirection = np.array(layersWindDirection)
        self._zenithInDeg = 0
        self._airmass = 1.0
        self._lambda = 500e-9
        self._checkSameShape(self._layersL0, self._layersJs, 'L0', 'Js')
        self._checkSameShape(self._layersAltitudeInMeterAtZenith,
                             self._layersJs, 'Altitude', 'Js')
        self._checkSameShape(self._layersWindSpeed, self._layersJs,
                             'WindSpeed', 'Js')
        self._checkSameShape(self._layersWindDirection, self._layersJs,
                             'WindDirection', 'Js')

    def _checkSameShape(self, a, b, nameA, nameB):
        assert a.shape == b.shape, '%s shape %s != %s shape %s' % (
            nameA, a.shape, nameB, b.shape)

    @classmethod
    def from_r0s(cls,
                 layersR0,
                 layersL0,
                 layersAltitude,
                 layersWindSpeed,
                 layersWindDirection):
        """
        Cn2 profile constructor from r0 values of each layer

        Parameters:
            layersR0 (ndarray): array of layers r0 in meters at 500nm
            layersL0 (ndarray): array of layers outer-scale L0 in meters
            layersAltitude (ndarry): array of layers altitude at Zenith in
                meters

        Every array must be 1D and have the same size, defining the number of
            layers of the profile
        All parameters defined at zenith
        """
        layersR0 = cls._quantitiesToValue(layersR0)
        layersL0 = cls._quantitiesToValue(layersL0)
        layersAltitude = cls._quantitiesToValue(layersAltitude)
        layersWindSpeed = cls._quantitiesToValue(layersWindSpeed)
        layersWindDirection = cls._quantitiesToValue(
            np.deg2rad(layersWindDirection))
        js = cls._r02js(layersR0, cls.DEFAULT_AIRMASS, cls.DEFAULT_LAMBDA)
        return Cn2Profile(js,
                          layersL0,
                          layersAltitude,
                          layersWindSpeed,
                          layersWindDirection)

    @classmethod
    def from_fractional_j(cls,
                          r0AtZenith,
                          layersFractionalJ,
                          layersL0,
                          layersAltitude,
                          layersWindSpeed,
                          layersWindDirection):
        """
        Cn2 profile constructor from total r0 at zenith and fractional J of
            each layer

        Parameters:
            r0AtZenith (float): overall r0 at zenith [m]
            layersFractionalJ (ndarray): array of J values for each layer.
                Array must sum up to 1
            layersL0 (ndarray): array of layers outer-scale L0 in meters
            layersAltitude (ndarry): array of layers altitude at Zenith in
                meters

        Every array must be 1D and have the same size, defining the number of
            layers of the profile
        All parameters defined at zenith
        """
        r0AtZenith = cls._quantitiesToValue(r0AtZenith)
        layersFractionalJ = cls._quantitiesToValue(layersFractionalJ)
        layersL0 = cls._quantitiesToValue(layersL0)
        layersAltitude = cls._quantitiesToValue(layersAltitude)
        layersWindSpeed = cls._quantitiesToValue(layersWindSpeed)
        layersWindDirection = cls._quantitiesToValue(
            np.deg2rad(layersWindDirection))
        sumJ = np.array(layersFractionalJ).sum()
        assert (np.abs(sumJ - 1.0) < 0.01), \
            "Total of J values must be 1.0, got %g" % sumJ
        totalJ = r0AtZenith ** (-5. / 3) / (
            0.422727 * (2 * np.pi / cls.DEFAULT_LAMBDA) ** 2 * 
            cls.DEFAULT_AIRMASS)
        js = np.array(layersFractionalJ) * totalJ
        return Cn2Profile(js,
                          layersL0,
                          layersAltitude,
                          layersWindSpeed,
                          layersWindDirection)

    @staticmethod
    def _quantitiesToValue(quantity):
        if isinstance(quantity, Iterable):
            if isinstance(quantity[0], Quantity):
                value = np.array(
                    [quantity[i].si.value for i in range(len(quantity))])
            else:
                value = value = np.array(
                    [quantity[i] for i in range(len(quantity))])
        else:
            if isinstance(quantity, Quantity):
                value = quantity.si.value
            else:
                value = quantity
        return value

    @staticmethod
    def _r02js(r0s, airmass, wavelength):
        js = np.array(r0s) ** (-5. / 3) / (
            0.422727 * airmass * (2 * np.pi / wavelength) ** 2)
        return js

    @staticmethod
    def _js2r0(Js, airmass, wavelength):
        r0s = (0.422727 * airmass * (2 * np.pi / wavelength) ** 2 * 
               np.array(Js)) ** (-3. / 5)
        return r0s

    def set_zenith_angle(self, zenithAngleInDeg):
        self._zenithInDeg = zenithAngleInDeg
        self._airmass = self.zenith_angle_to_airmass(zenithAngleInDeg)

    def zenith_angle(self):
        return self._zenithInDeg * u.deg

    @staticmethod
    def zenith_angle_to_airmass(zenithAngleInDeg):
        zenithInRad = zenithAngleInDeg * Constants.DEG2RAD
        return 1. / np.cos(zenithInRad)

    def airmass(self):
        return self._airmass * u.dimensionless_unscaled

    def layers_distance(self):
        return self._layersAltitudeInMeterAtZenith * self._airmass * u.m

    def set_wavelength(self, wavelengthInMeters):
        self._lambda = wavelengthInMeters

    def wavelength(self):
        return self._lambda * u.m

    def number_of_layers(self):
        return Quantity(len(self._layersJs), dtype=int)

    def wind_speed(self):
        """
        Returns:
            (array): windspeed of each layer in m/s
        """
        return self._layersWindSpeed * u.m / u.s

    def wind_direction(self):
        """
        Returns:
            (array): wind direction of each layer [deg, clockwise from N]
        """
        # TODO: check the unit. layersWindDirection is converted in SI, so
        # in radiants. I think that I must return np.rad2deg(layersWindDir) *
        # u.deg
        return np.rad2deg(self._layersWindDirection) * u.deg

    def seeing(self):
        """
        Returns:
            seeing value in arcsec at specified lambda and zenith angle
            defined as 0.98 * lambda / r0
        """
        return 0.98 * self._lambda / self.r0() * Constants.RAD2ARCSEC * \
            u.arcsec

    def r0(self):
        '''
        Returns:
            r0 (float): Fried parameter at specified lambda and zenith angle
        '''
        return (0.422727 * self._airmass * 
                (2 * np.pi / self._lambda) ** 2 * 
                np.sum(self._layersJs)) ** (-3. / 5) * u.m

    def r0s(self):
        '''
        Returns:
            r0s (numpy array): Fried parameter at specified lambda
            and zenith angle for each layer
        '''
        return self._js2r0(self._layersJs,
                           self._airmass,
                           self._lambda) * u.m

    def theta0(self):
        '''
        Returns:
            theta0 (float): isoplanatic angle at specified lambda and zenith
                angle [arcsec]
        '''
        return (2.914 * self._airmass ** (8. / 3) * 
                (2 * np.pi / self._lambda) ** 2 * 
                np.sum(self._layersJs * 
                       self._layersAltitudeInMeterAtZenith ** (5. / 3))
                ) ** (-3. / 5) * Constants.RAD2ARCSEC * u.arcsec

    def tau0(self):
        '''
        Returns:
            tau0 (float): tau0 at specified lambda and zenith angle [sec]
        '''
        return (2.914 * self._airmass * 
                (2 * np.pi / self._lambda) ** 2 * 
                np.sum(self._layersJs * 
                       self._layersWindSpeed ** (5. / 3))
                ) ** (-3. / 5) * u.s

    def outer_scale(self):
        return self._layersL0 * u.m


class MaorySteroScidarProfiles():

    def __init__(self):
        pass

    @staticmethod
    def _retrieveMaoryP():
        rootDir = dataRootDir()
        filename = os.path.join(rootDir,
                                'cn2profiles',
                                'maoryPProfiles.fits')
        return fits.getdata(filename)

    @classmethod
    def _profileMaker(cls, idxJ, idxW):
        rr = cls._retrieveMaoryP()
        h = rr[0] * 1e3
        js = rr[idxJ]
        L0s = np.ones(len(js)) * 30.
        windSpeed = rr[idxW]
        windDirection = np.linspace(0, 360, len(js))
        return Cn2Profile(js, L0s, h, windSpeed, windDirection)

    @classmethod
    def P10(cls):
        return cls._profileMaker(1, 6)

    @classmethod
    def P25(cls):
        return cls._profileMaker(2, 7)

    @classmethod
    def P50(cls):
        return cls._profileMaker(3, 8)

    @classmethod
    def P75(cls):
        return cls._profileMaker(4, 9)

    @classmethod
    def P90(cls):
        return cls._profileMaker(5, 10)


class EsoEltProfiles():

    L0 = 25

    @staticmethod
    def _restoreProfiles():
        rootDir = dataRootDir()
        filename = os.path.join(rootDir,
                                'cn2profiles',
                                'cn2_eso_258292_2.fits')
        return fits.getdata(filename, header=True)

    @classmethod
    def _profileMaker(cls, keyR0, idxJ, idxWind):
        rr, hdr = cls._restoreProfiles()
        r0 = hdr[keyR0]
        h = rr[0]
        js = rr[idxJ] / 100
        windSpeed = rr[idxWind]
        windDirection = np.linspace(0, 360, len(js))
        L0s = np.ones(len(js)) * cls.L0
        return Cn2Profile.from_fractional_j(
            r0, js, L0s, h, windSpeed, windDirection)

    @classmethod
    def Median(cls):
        return cls._profileMaker('R0MEDIAN', 1, 6)

    @classmethod
    def Q1(cls):
        return cls._profileMaker('R0Q1', 2, 7)

    @classmethod
    def Q2(cls):
        return cls._profileMaker('R0Q2', 3, 8)

    @classmethod
    def Q3(cls):
        return cls._profileMaker('R0Q3', 4, 9)

    @classmethod
    def Q4(cls):
        return cls._profileMaker('R0Q4', 5, 10)


class MiscellaneusProfiles():

    @classmethod
    def MaunaKea(cls):
        '''
        From Brent L. Ellerbroek, Francois J. Rigaut,
        "Scaling multiconjugate adaptive optics performance estimates
        to extremely large telescopes,"
        Proc. SPIE 4007, Adaptive Optical Systems Technology,
        (7 July 2000); doi: 10.1117/12.390314
        '''
        r0 = 0.236
        hs = np.array([0.09, 1.826, 2.72, 4.256, 6.269, 8.34,
                      10.546, 12.375, 14.61, 16.471, 17.028]) * 1000
        js = [0.003, 0.136, 0.163, 0.161, 0.167,
              0.234, 0.068, 0.032, 0.023, 0.006, 0.007]
        windSpeed = np.ones(len(js)) * 10.0
        windDirection = np.linspace(0, 360, len(js))
        L0s = np.ones(len(js)) * 25.0
        return Cn2Profile.from_fractional_j(
            r0, js, L0s, hs, windSpeed, windDirection)
