'''
@author: lbusoni
'''
import numpy as np
from apposto.utils.constants import Constants
from apposto.utils.package_data import dataRootDir
import os
from astropy.io import fits

"""

k = 2pi/lambda
J[i] = dh[i] * Cn2[i]
r0[i] = (0.422727 * airmass * k**2 J[i]) ** (-3/5)

r0 = (Sum r0[i]**(-5/3) ) ** (-3/5) = (Sum 0.423 airmass k**2 J[i]) ** (-3/5)
theta0 = (2.914 k**2 airmass**(8/3) Sum(J*h**(5/3))) ** (-3/5)
"""


class Cn2Profile(object):

    DEFAULT_LAMBDA=500e-9
    DEFAULT_AIRMASS=1.0

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
            layersAltitude (ndarry): array of layers altitude at Zenith in meters

        Every array must be 1D and have the same size, defining the number of layers of the profile
        All parameters defined at zenith
        """
        self._layersJs= np.array(layersJs)
        self._layersL0= np.array(layersL0)
        self._layersAltitudeInMeterAtZenith= np.array(layersAltitude)
        self._layersWindSpeed= np.array(layersWindSpeed)
        self._layersWindDirection= np.array(layersWindDirection)
        self._zenithInDeg= 0
        self._airmass= 1.0
        self._lambda= 500e-9
        assert True


    @classmethod
    def fromR0s(cls,
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
            layersAltitude (ndarry): array of layers altitude at Zenith in meters

        Every array must be 1D and have the same size, defining the number of layers of the profile
        All parameters defined at zenith
        """
        js= cls._r0ToJs(layersR0, cls.DEFAULT_AIRMASS, cls.DEFAULT_LAMBDA)
        return Cn2Profile(js,
                          layersL0,
                          layersAltitude,
                          layersWindSpeed,
                          layersWindDirection)


    @classmethod
    def fromFractionalJ(cls,
                        r0AtZenith,
                        layersFractionalJ,
                        layersL0,
                        layersAltitude,
                        layersWindSpeed,
                        layersWindDirection):
        """
        Cn2 profile constructor from total r0 at zenith and fractional J of each layer

        Parameters:
            r0AtZenith (float): overall r0 at zenith [m]
            layersFractionalJ (ndarray): array of J values for each layer. Array must sum up to 1
            layersL0 (ndarray): array of layers outer-scale L0 in meters
            layersAltitude (ndarry): array of layers altitude at Zenith in meters

        Every array must be 1D and have the same size, defining the number of layers of the profile
        All parameters defined at zenith
        """
        sumJ= np.array(layersFractionalJ).sum()
        assert (np.abs(sumJ - 1.0) < 0.01), \
            "Total of J values must be 1.0, got %g" % sumJ
        totalJ= r0AtZenith ** (-5./3) / (
            0.422727 * (2*np.pi/cls.DEFAULT_LAMBDA)**2 * 
            cls.DEFAULT_AIRMASS)
        js= np.array(layersFractionalJ) * totalJ
        return Cn2Profile(js,
                          layersL0,
                          layersAltitude,
                          layersWindSpeed,
                          layersWindDirection)


    @staticmethod
    def _r0ToJs(r0s, airmass, wavelength):
        js= np.array(r0s) ** (-5./3) / (
            0.422727 * airmass * (2*np.pi/wavelength)**2)
        return js


    @staticmethod
    def _jsToR0(Js, airmass, wavelength):
        r0s = (0.422727 * airmass * (2*np.pi/wavelength)**2 *
               np.array(Js)) ** (-3./5)
        return r0s


    def setZenithAngle(self, zenithAngleInDeg):
        self._zenithInDeg= zenithAngleInDeg
        self._airmass= self.zenithAngle2Airmass(zenithAngleInDeg)


    def zenithAngle(self):
        return self._zenithInDeg


    @staticmethod
    def zenithAngle2Airmass(zenithAngleInDeg):
        zenithInRad= zenithAngleInDeg * Constants.DEG2RAD
        return 1./ np.cos(zenithInRad)


    def airmass(self):
        return self._airmass

    def layersDistance(self):
        return self._layersAltitudeInMeterAtZenith * self._airmass


    def setWavelength(self, wavelengthInMeters):
        self._lambda = wavelengthInMeters


    def wavelength(self):
        return self._lambda


    def numberOfLayers(self):
        return len(self._layersJs)

    def windSpeed(self):
        """
        Returns:
            (array): windspeed of each layer in m/s
        """
        return self._layersWindSpeed

    def windDirection(self):
        """
        Returns:
            (array): wind direction of each layer [deg, clockwise from N]
        """
        return self._layersWindDirection

    def seeing(self):
        """
        Returns:
            seeing value in arcsec at specified lambda and zenith angle
            defined as 0.98 * lambda / r0
        """
        return 0.98 * self._lambda / self.r0() * Constants.RAD2ARCSEC


    def r0(self):
        '''
        Returns:
            r0 (float): Fried parameter at specified lambda and zenith angle
        '''
        return (0.422727 * self.airmass() *
                (2*np.pi/self.wavelength())**2 *
                np.sum(self._layersJs))**(-3./5)



    def theta0(self):
        '''
        Returns:
            theta0 (float): isoplanatic angle at specified lambda and zenith angle [arcsec]
        '''
        return (2.914 * self.airmass()**(8./3) *
                (2*np.pi/self.wavelength())**2 *
                np.sum(self._layersJs*
                       self._layersAltitudeInMeterAtZenith**(5./3))
                )**(-3./5) * Constants.RAD2ARCSEC


    def tau0(self):
        '''
        Returns:
            tau0 (float): tau0 at specified lambda and zenith angle [sec]
        '''
        return (2.914 * self.airmass() *
                (2*np.pi/self.wavelength())**2 *
                np.sum(self._layersJs*
                       self._layersWindSpeed**(5./3))
                )**(-3./5)


class MaoryProfiles():

    def __init__(self):
        pass

    @staticmethod
    def _retrieveMaoryP():
        rootDir= dataRootDir()
        filename= os.path.join(rootDir,
                               'cn2profiles',
                               'maoryPProfiles.fits')
        return fits.getdata(filename)

    @classmethod
    def P10(cls):
        rr= cls._retrieveMaoryP()
        h= rr[0]*1e3
        js= rr[1]
        L0= 30.
        windSpeed= rr[6]
        windDirection= np.linspace(0, 360, len(js))
        return Cn2Profile(js, L0, h, windSpeed, windDirection)

    @classmethod
    def P25(cls):
        rr= cls._retrieveMaoryP()
        h= rr[0]*1e3
        js= rr[2]
        L0= 30.
        windSpeed= rr[7]
        windDirection= np.linspace(0, 360, len(js))
        return Cn2Profile(js, L0, h, windSpeed, windDirection)

    @classmethod
    def P50(cls):
        rr= cls._retrieveMaoryP()
        h= rr[0]*1e3
        js= rr[3]
        L0= 30.
        windSpeed= rr[8]
        windDirection= np.linspace(0, 360, len(js))
        return Cn2Profile(js, L0, h, windSpeed, windDirection)

    @classmethod
    def P75(cls):
        rr= cls._retrieveMaoryP()
        h= rr[0]*1e3
        js= rr[4]
        L0= 30.
        windSpeed= rr[9]
        windDirection= np.linspace(0, 360, len(js))
        return Cn2Profile(js, L0, h, windSpeed, windDirection)

    @classmethod
    def P90(cls):
        rr= cls._retrieveMaoryP()
        h= rr[0]*1e3
        js= rr[5]
        L0= 30.
        windSpeed= rr[10]
        windDirection= np.linspace(0, 360, len(js))
        return Cn2Profile(js, L0, h, windSpeed, windDirection)
