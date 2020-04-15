import numpy as np
from arte.utils.constants import Constants

__version__= "$Id: $"


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
