import numpy as np
from arte.types.region_of_interest import RegionOfInterest


class CircularMask():

    def __init__(self,
                 frameShape,
                 maskRadius=None,
                 maskCenter=None):
        self._shape = frameShape
        self._maskRadius = maskRadius
        self._maskCenter = maskCenter
        self._mask = None
        self._computeMask()

    def __repr__(self):
        return "shape %s, radius %f, center %s" % (
            self._shape, self._maskRadius, self._maskCenter)

    def _computeMask(self):
        if self._maskRadius is None:
            self._maskRadius = min(self._shape) / 2.
        if self._maskCenter is None:
            self._maskCenter = 0.5 * np.array([self._shape[0],
                                               self._shape[1]])

        r = self._maskRadius
        cc = self._maskCenter
        y, x = np.mgrid[0.5: self._shape[0] + 0.5:1,
                        0.5: self._shape[1] + 0.5:1]
        self._mask = np.where(
            ((x - cc[1])**2 + (y - cc[0])**2) <= r**2, False, True)

    def mask(self):
        return self._mask

    def asTransmissionValue(self):
        return np.logical_not(self._mask).astype(np.int)

    def radius(self):
        return self._maskRadius

    def center(self):
        return self._maskCenter

    def shape(self):
        return self._shape

    @staticmethod
    def fromMaskedArray(maskedArray):
        assert isinstance(maskedArray, np.ma.masked_array)
        shape = maskedArray.shape
        pointsMasked = np.argwhere(maskedArray.mask == False)
        diameters = pointsMasked.max(axis=0) - pointsMasked.min(axis=0) + 1
        assert (diameters[0] == diameters[1]), \
            'x-diameter and y-diameter differ (%s px)' % str(diameters)
        radius = 0.5 * diameters[0]
        center = 0.5 * (pointsMasked.max(axis=0) +
                        pointsMasked.min(axis=0) + 1)
        circularMask = CircularMask(shape, radius, center)
        assert (circularMask.mask() == maskedArray.mask).all(), \
            'the mask of the maskedArray is not circular'
        return circularMask

    def regionOfInterest(self):
        centerX = int(self.center()[1])
        centerY = int(self.center()[0])
        radius = int(self.radius())
        return RegionOfInterest(centerX - radius, centerX + radius,
                                centerY - radius, centerY + radius)
