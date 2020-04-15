from astropy import units
from arte.utils.constants import Constants


class Seeing(object):


    def __init__(self, seeingAt500nm):
        self._seeingAt500nm= seeingAt500nm
        self._r0At500nm= 500*units.nm / (
            seeingAt500nm * Constants.ARCSEC2RAD)

    def toR0(self, wavelength=500*units.nm):
        return self._r0At500nm*(wavelength / 500*units.nm)**(6./5)
