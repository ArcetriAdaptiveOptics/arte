import numpy as np
from arte.types.region_of_interest import RegionOfInterest
from arte.utils.image_moments import ImageMoments

# class BaseMask


class CircularMask():
    '''
    Represent a circular mask

    A `~numpy.array` representing a circular pupil. Frame shape, pupil radius
    and center can be specified.

    Use `mask` method to access the mask as boolean mask (e.g. to be used
    in a `~numpy.ma.masked_array` object) with False values where the frame is 
    not masked (i.e. within the pupil) and True values outside

    Use `asTransmissionValue` method to acces the mask as transmission mask, i.e. 
    1 within the pupil and 0 outside. Fractional transmission for edge pixels
    is not implemented

    If a `~numpy.ma.masked_array` having a circular mask is available, the static
    method `fromMaskedArray` can be used to create a `CircularMask` object having
    the same shape of the masked array and the same pupil center and radius


    Parameters
    ----------
        frameShape: tuple (2,)
            shape of the returned array

        maskRadius: real
            pupil radius in pixel

        maskCenter: list (2,) or `~numpy.array`
            Y-X coordinates of the pupil center in pixel
    '''

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
        '''
        Boolean mask of the Circular Mask

        Returns
        -------
        mask: boolean `~numpy.array`
            mask of the circular pupil. Array is True outside the pupil,
            and False inside the pupil
        '''
        return self._mask

    def asTransmissionValue(self):
        return np.logical_not(self._mask).astype(int)

    def radius(self):
        '''
        Radius of the mask 

        Returns
        -------
        radius: real
            mask radius in pixel

        '''
        return self._maskRadius

    def center(self):
        '''
        Y, X coordinates of the mask center

        Returns
        -------
        center: `~numpy.array` of shape (2,)
            Y, X coordinate of the mask center in the array reference system

        '''
        return self._maskCenter

    def shape(self):
        '''
        Array shape

        Returns
        -------
        shape: list (2,)
            shape of the mask array
        '''
        return self._shape

    @staticmethod
    def fromMaskedArray(maskedArray):
        '''
        Creates a `CircularMask` roughly corresponding to the mask of a masked
        array
        
        Returns a `CircularMask` object having radius and center guessed from
        the passed mask using `ImageMoments` centroid and semiAxes.
        Important note: the created `CircularMask` object 
        is not guaranteed to have the same set of valid points of the 
        passed mask, but it is included in the passed mask, i.e. all the 
        valid points of the created mask are also valid points of the passed 
        masked array 
        
        Parameters
        ----------
        maskedArray: `~numpy.ma.MaskedArray` 
            a masked array with a circular mask

        Returns
        -------
        circular_mask: `CircularMask`
            a circular mask included in the mask of `maskedArray`

        '''
        assert isinstance(maskedArray, np.ma.masked_array)
        shape = maskedArray.shape
        again = 0.995
        while again:
            im = ImageMoments(maskedArray.mask.astype(int)*-1 + 1)
            centerYX = np.roll(im.centroid(),1)
            radius = again*np.min(im.semiAxes())
            circularMask = CircularMask(shape, radius, centerYX)
            if np.in1d(circularMask.in_mask_indices(),
                       np.argwhere(maskedArray.mask.flatten() == False)).all():
                again = False
            if radius < 1:
                raise Exception("Couldn't estimate a CircularMask")
            else:
                again *= 0.9

        return circularMask

    def regionOfInterest(self):
        centerX = int(self.center()[1])
        centerY = int(self.center()[0])
        radius = int(self.radius())
        return RegionOfInterest(centerX - radius, centerX + radius,
                                centerY - radius, centerY + radius)

    def as_masked_array(self):
        return np.ma.array(np.ones(self._shape),
                           mask=self.mask())

    def in_mask_indices(self):
        return self.asTransmissionValue().flatten().nonzero()[0]

class AnnularMask(CircularMask):
    '''
    Inheritance of CircularMask class to provide an annular mask

    Added inRadius parameter, radius of central obstruction.
    Default inRadius values is 0 with AnnularMask converging to CircularMask
    '''

    def __init__(self,
                 frameShape,
                 maskRadius=None,
                 maskCenter=None,
                 inRadius=0):
        self._inRadius = inRadius
        super().__init__(frameShape, maskRadius, maskCenter)

    def __repr__(self):
        return "shape %s, radius %f, center %s, inradius %f" % (
            self._shape, self._maskRadius, self._maskCenter, self._inRadius)

    def inRadius(self):
        return self._inRadius

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

        tmp = ((x - cc[1])**2 + (y - cc[0])**2) <= r**2
        if self._inRadius == 0:
            self._mask = np.where(tmp, False, True)
        else:
            cc = CircularMask(self._shape, self._inRadius, self._maskCenter)
            tmp[cc.asTransmissionValue() > 0] = False
            self._mask = np.where(tmp, False, True)

    @staticmethod
    def fromMaskedArray(maskedArray):
        raise Exception("Not implemented yet. ")
