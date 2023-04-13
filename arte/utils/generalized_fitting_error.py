import numpy as np
from arte.utils.constants import DEG2RAD, ARCSEC2RAD

__version__ = "$Id: $"


class GeneralizedFittingError(object):

    def __init__(self,
                 spatialFrequenciesInInverseMeter,
                 altitudeOfDeformableMirrorsInMeter,
                 pitchOfDeformableMirrorsInMeter,
                 optimizedFieldOfViewInArcsec,
                 cn2Profile,
                 zenithAngleInDeg,
                 seeingInArcsec):
        self._spat_freqs = spatialFrequenciesInInverseMeter
        self._hdm = altitudeOfDeformableMirrorsInMeter
        self._pitch = pitchOfDeformableMirrorsInMeter
        self._fov = optimizedFieldOfViewInArcsec
        self._cn2 = cn2Profile
        self._zenithInRad = zenithAngleInDeg * DEG2RAD
        self._seeingInRad = seeingInArcsec * ARCSEC2RAD

        airmass = 1. / np.cos(self._zenithInRad)
