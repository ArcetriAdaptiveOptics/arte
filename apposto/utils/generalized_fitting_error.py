import numpy as np
from apposto.utils.constants import Constants

__version__= "$Id: $"


class Cn2Profile(object):


    def __init__(self, layersAltitudeInMeterAtZenith, cn2Profile):
        self._layersAltitudeInMeterAtZenith= \
            layersAltitudeInMeterAtZenith
        self._cn2= cn2Profile
        self._zenithInRad= 0
        self._airmass= 1.0


    def setZenithAngle(self, zenithAngleInDeg):
        self._zenithInRad= zenithAngleInDeg * Constants.DEG2RAD
        self._airmass= 1./ np.cos(self._zenithInRad)


    def layersDistance(self):
        return self._layersAltitudeInMeterAtZenith * self._airmass


class GeneralizedFittingError(object):

    def __init__(self,
                 spatialFrequenciesInInverseMeter,
                 altitudeOfDeformableMirrorsInMeter,
                 pitchOfDeformableMirrorsInMeter,
                 optimizedFieldOfViewInArcsec,
                 cn2Profile,
                 zenithAngleInDeg,
                 seeingInArcsec):
        self._freqs= spatialFrequenciesInInverseMeter
        self._hdm= altitudeOfDeformableMirrorsInMeter
        self._pitch= pitchOfDeformableMirrorsInMeter
        self._fov= optimizedFieldOfViewInArcsec
        self._cn2= cn2Profile
        self._zenithInRad= zenithAngleInDeg * Constants.DEG2RAD
        self._seeingInRad= seeingInArcsec * Constants.ARCSEC2RAD

        airmass= 1./np.cos(self._zenithInRad)
