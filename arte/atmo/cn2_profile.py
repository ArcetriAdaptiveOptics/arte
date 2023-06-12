r''' This module manages atmospheric Cn2 profiles.

The class Cn2Profile is the base class: it allows to describe a profile
as a set of layers each one with specified J, outer scale :math:`L_0`,
wind speed and wind direction and located at a specified altitude h
above the ground

Cn2Profile allows to specify the zenith angle z and the wavelength and 

The following relations are used, where the index i identifies
the i-th layer and the airmass is :math:`X=\sec z`.


.. math::
   :nowrap:

   \begin{eqnarray}
    k & = & \frac{2 \pi}{\lambda} \\
    J_i & = & {C_n^2}_i \, dh_i  \\
    {r_0}_i & = & (0.423 \, X \, k^2 J_i)^{-3/5} \\
    r_0 & = & \big( \sum{{{r_0}_i}^{-5/3}} \big)^{-3/5} \\
        & = & \big( \sum{0.423 X k^2 J_i} \big)^{-3/5} \\
    \theta_0 & = & \big( 2.914 k^2 X^{8/3} \sum{J_i h_i^{5/3}} \big)^{-3/5} 
   \end{eqnarray}

'''
import numpy as np
from astropy.units.quantity import Quantity
from arte.utils.constants import DEG2RAD, RAD2ARCSEC
from arte.utils.package_data import dataRootDir
import os
from astropy.io import fits
import astropy.units as u
from astropy.utils.misc import isiterable


class Cn2Profile(object):
    """
    Parameters
    ----------
        layersJs: :class:`~numpy:numpy.ndarray`
                  array of layers Js in meters**(1/3)
        layersL0: :class:`~numpy:numpy.ndarray`
                  array of layers outer-scale L0 in meters
        layersAltitude: :class:`~numpy:numpy.ndarray`
                  array of layers altitude at Zenith in meters
        layersWindSpeed: :class:`~numpy:numpy.ndarray`
                  array of layers wind speed in meters per second
        layersWindDirection: :class:`~numpy:numpy.ndarray`
                  array of layers wind direction in degrees, clockwise
                  from North

    Every array must be 1D and have the same size, defining the number of
    layers of the profile

    All parameters must be defined at zenith
    """

    DEFAULT_LAMBDA = 500e-9
    DEFAULT_AIRMASS = 1.0

    def __init__(self,
                 layersJs,
                 layersL0,
                 layersAltitude,
                 layersWindSpeed,
                 layersWindDirection):
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

        Parameters
        ----------
            layersR0: :class:`~numpy:numpy.ndarray`
                      array of layers r0 in meters at 500nm
            layersL0: :class:`~numpy:numpy.ndarray`
                      array of layers outer-scale L0 in meters
            layersAltitude: :class:`~numpy:numpy.ndarray`
                      array of layers altitude at Zenith in meters
            layersWindSpeed: :class:`~numpy:numpy.ndarray`
                      array of layers wind speed in meters per second
            layersWindDirection: :class:`~numpy:numpy.ndarray`
                      array of layers wind direction in degrees, clockwise
                      from North


        Every array must be 1D and have the same size, defining the number of
        layers of the profile

        All parameters must be defined at zenith
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

        Parameters
        ----------
            r0AtZenith: float
                        overall r0 at zenith [m]
            layersFractionalJ: :class:`~numpy:numpy.ndarray`
                               array of J values for each layer.
                               Array must sum up to 1
            layersL0: :class:`~numpy:numpy.ndarray`
                      array of layers outer-scale L0 in meters
            layersAltitude: :class:`~numpy:numpy.ndarray`
                      array of layers altitude at Zenith in meters
            layersWindSpeed: :class:`~numpy:numpy.ndarray`
                      array of layers wind speed in meters per second
            layersWindDirection: :class:`~numpy:numpy.ndarray`
                      array of layers wind direction in degrees, clockwise
                      from North


        Every array must be 1D and have the same size, defining the number of
        layers of the profile

        All parameters must be defined at zenith

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
        if isiterable(quantity):
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
        if isinstance(zenithAngleInDeg, Quantity):
            value = zenithAngleInDeg.to(u.deg).value
        else:
            value = zenithAngleInDeg
        self._zenithInDeg = value
        self._airmass = self.zenith_angle_to_airmass(value)

    def zenith_angle(self):
        return self._zenithInDeg * u.deg

    @staticmethod
    def zenith_angle_to_airmass(zenithAngleInDeg):
        if isinstance(zenithAngleInDeg, Quantity):
            value = zenithAngleInDeg.to(u.deg).value
        else:
            value = zenithAngleInDeg
        zenithInRad = value * DEG2RAD
        return 1. / np.cos(zenithInRad)

    def airmass(self):
        '''
        Returns
        -------
        airmass: float
            airmass at specified zenith angle
        '''
        return self._airmass * u.dimensionless_unscaled

    def layers_distance(self):
        return self._layersAltitudeInMeterAtZenith * self._airmass * u.m

    def set_wavelength(self, wavelengthInMeters):
        if isinstance(wavelengthInMeters, Quantity):
            value = wavelengthInMeters.to(u.m).value
        else:
            value = wavelengthInMeters
        self._lambda = value

    def wavelength(self):
        return self._lambda * u.m

    def number_of_layers(self):
        return Quantity(len(self._layersJs), dtype=int)

    def fractional_j(self):
        return Quantity(self._layersJs)

    def set_wind_speed(self, windSpeed):
        if isinstance(windSpeed, Quantity):
            value = windSpeed.to(u.m / u.s).value
        else:
            value = windSpeed
        self._layersWindSpeed = value

    def wind_speed(self):
        """
        Returns:
            (array): windspeed of each layer in m/s
        """
        return self._layersWindSpeed * u.m / u.s

    def mean_wind_speed(self):
        return (
            np.sum(
                self._layersWindSpeed**(5. / 3) * self._layersJs
            ) / np.sum(
                self._layersJs)
        )**(3. / 5) * u.m / u.s

    def set_wind_direction(self, windDirectionInDeg):
        if isinstance(windDirectionInDeg, Quantity):
            value = windDirectionInDeg.to(u.rad).value
        else:
            value = np.deg2rad(windDirectionInDeg)
        self._layersWindDirection = value

    def wind_direction(self):
        """
        Returns
        -------
        wind: :class:`~astropy:astropy.units.quantity.Quantity` containing an array.
            wind direction of each layer [deg, clockwise from N]
        """
        # TODO: check the unit. layersWindDirection is converted in SI, so
        # in radiants. I think that I must return np.rad2deg(layersWindDir) *
        # u.deg
        return np.rad2deg(self._layersWindDirection) * u.deg

    def seeing(self):
        """
        Returns
        -------
        seeing: :class:`~astropy:astropy.units.quantity.Quantity` equivalent to arcsec
            seeing value at specified lambda and zenith angle
            defined as 0.98 * lambda / r0
        """
        return 0.98 * self.wavelength() / self.r0() * RAD2ARCSEC * \
            u.arcsec

    def r0(self):
        '''
        Returns
        -------
        r0: :class:`~astropy:astropy.units.quantity.Quantity` equivalent to meters
            Fried parameter at defined wavelength and zenith angle
        '''
        return (0.422727 * self._airmass *
                (2 * np.pi / self._lambda) ** 2 *
                np.sum(self._layersJs)) ** (-3. / 5) * u.m

    def r0s(self):
        '''
        Returns
        -------
        r0: :class:`~astropy:astropy.units.quantity.Quantity` equivalent to meters of :class:`~numpy:numpy.ndarray`
            Fried parameter of each layer at defined wavelength and zenith angle 
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
                ) ** (-3. / 5) * RAD2ARCSEC * u.arcsec

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


class MaoryStereoScidarProfiles2021():

    L0 = 25

    @staticmethod
    def _restoreMaoryProfiles():
        rootDir = dataRootDir()
        filename = os.path.join(rootDir,
                                'cn2profiles',
                                'referenceProfiles35layers_new.fits')
        return fits.getdata(filename)

    @classmethod
    def _profileMaker(cls, idxJ, idxW):
        rr = cls._restoreMaoryProfiles()
        h = rr[0] * 1e3
        js = rr[idxJ]
        L0s = np.ones(len(js)) * cls.L0
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
    def _profileMaker(cls, keyR0, idxJ, idxWind, L0=None):
        rr, hdr = cls._restoreProfiles()
        r0 = hdr[keyR0]
        h = rr[0]
        js = rr[idxJ] / 100
        windSpeed = rr[idxWind]
        windDirection = np.linspace(0, 360, len(js))
#         windDirection = np.random.uniform(0, 360, len(js))
        if L0 is None:
            L0 = cls.L0
        L0s = np.ones(len(js)) * L0
        return Cn2Profile.from_fractional_j(
            r0, js, L0s, h, windSpeed, windDirection)

    @classmethod
    def Median(cls, *args, **kwargs):
        return cls._profileMaker('R0MEDIAN', 1, 6, *args, **kwargs)

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

    @classmethod
    def LBT(cls):
        '''
        From G. Agapito, C. Arcidiacono, F. Quiros-Pacheco, S. Esposito,
        "Adaptive optics at short wavelengths - Expected performance and sky
        coverage of the FLAO system going toward visible wavelengths",
        doi: 10.1007/s10686-014-9380-7
        '''
        r0 = 0.140
        hs = np.array([0.103, 0.725, 2.637, 11.068]) * 1000
        js = [0.70, 0.06, 0.14, 0.10]
        windSpeed = [2., 4., 6., 25.]
        windDirection = [0., 90., 180., 270.]
        L0s = np.ones(len(js)) * 40.0
        return Cn2Profile.from_fractional_j(
            r0, js, L0s, hs, windSpeed, windDirection)

    @classmethod
    def ERIS(cls):
        '''
        From Doc. No.: VLT-SPE-ESO-11250-4110 
        '''
        r0 = 0.12633
        hs = np.array([0.03, 0.140, 0.281, 0.562, 1.125, 2.250, 4.500,
                       7.750, 11.000, 14.000]) * 1000
        js = [0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04, 0.05, 0.05]
        windSpeed = [6.6, 5.9, 5.1, 4.5, 5.1, 8.3, 16.3, 30.2, 34.3, 17.5]
        windDirection = np.linspace(0, 360, len(js))
        L0s = np.ones(len(js)) * 22.
        return Cn2Profile.from_fractional_j(
            r0, js, L0s, hs, windSpeed, windDirection)
