import numpy as np
from arte.types.region_of_interest import RegionOfInterest
from arte.utils.image_moments import ImageMoments


class BaseMask():
    '''
    '''

    def __init__(self, mask_array):
        self._shape = mask_array.shape
        self._mask = mask_array
    
    def mask(self):
        '''
        Boolean mask of the mask

        Returns
        -------
        mask: boolean `~numpy.array`
            mask of the pupil. Array is True outside the pupil,
            and False inside the pupil
        '''
        return self._mask
    
    def as_masked_array(self):
        return np.ma.array(np.ones(self._shape),
                           mask=self.mask())
        
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
    def from_masked_array(numpy_masked_array):
        return BaseMask(numpy_masked_array)
    
    # TODO: funzione per verificare che tutti i punti di questa maschera stanno
    # dentro una maschera passata


class CircularMask(BaseMask):
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
            ((x - cc[1]) ** 2 + (y - cc[0]) ** 2) <= r ** 2, False, True)

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

    @staticmethod
    def fromMaskedArray(maskedArray, method='ImageMoments'):
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
        if method == "ImageMoments":
            again = 0.995
            while again:
                im = ImageMoments(maskedArray.mask.astype(int) * -1 + 1)
                centerYX = np.roll(im.centroid(), 1)
                radius = again * np.min(im.semiAxes())
                circularMask = CircularMask(shape, radius, centerYX)
                if np.in1d(circularMask.in_mask_indices(),
                           np.argwhere(maskedArray.mask.flatten() == False)).all():
                    again = False
                if radius < 1:
                    raise Exception("Couldn't estimate a CircularMask")
                else:
                    again *= 0.9
        elif method == "RANSAC":
            pass
        elif method == 'cog':
            img = np.asarray(self._mask.copy(), dtype=int)
            regions = measure.regionprops(img)
            bubble = regions[0]
            y0, x0 = bubble.centroid
            r = bubble.major_axis_length / 2.
            return [x0, y0, r]
        elif method == 'correlation':
            obj = sf.ShapeFitter(self._mask)
            obj.fit()
            return obj.params()
        elif method == "correlation":
            pass

        return circularMask

    def regionOfInterest(self):
        centerX = int(self.center()[1])
        centerY = int(self.center()[0])
        radius = int(self.radius())
        return RegionOfInterest(centerX - radius, centerX + radius,
                                centerY - radius, centerY + radius)

    def in_mask_indices(self):
        return self.asTransmissionValue().flatten().nonzero()[0]


    def _fit_circle_ransac(self,
                          apply_canny=True,
                          sigma=3,
                          display=False,
                          **keywords):
        '''Perform a circle fitting on the current mask using RANSAC algorithm

        Parameters
        ----------
            apply_canny: bool, default=True
                apply Canny edge detection before performing the fit.
            sigma: float, default=10
                if apply_canny is True, you can decide the Canny kernel size.
            display: bool, default=False
                it shows the result of the fit.
        '''
        self._shape_fitted = 'circle'
        self._method = 'ransac'
        img = np.asarray(self._mask.copy(), dtype=float)
        img[img > 0] = 128

        edge = img.copy()
        if apply_canny:
            edge = feature.canny(img, sigma)

        coords = np.column_stack(np.nonzero(edge))

        model, inliers = measure.ransac(
            coords, measure.CircleModel,
            keywords.pop('min_samples', 10), residual_threshold=0.01,
            max_trials=1000)
        cx, cy, r = model.params

        if display is True:
            print(r"Cx-Cy {:.2f}-{:.2f}, R {:.2f}".format(cx, cy, r))
            rr, cc = draw.disk((model.params[0], model.params[1]),
                               model.params[2],
                               shape=img.shape)
            img[rr, cc] += 512
            # plt.figure()
            self._dispnobl(img)

        self._params = model.params
        self._success = model.estimate(coords)

    def _fit_circle_correlation(self,
                               method='Nelder-Mead',
                               display=False,
                               **keywords):
        '''Perform a circle fitting on the current mask using minimization
        algorithm  with correlation merit functions.

        Tested with following minimizations methods: 'Nelder-Mead'. Relative
        precision of 1% reached on synthetic images without noise.

        Parameters
        ----------
            method: string, default='Nelder-Mead'
                from scipy.optimize.minimize.
            display: bool, default=False
                SLOWLY shows the progress of the fit.
            **keywords: dict, optional
                passed to scipy.optimize.minimize
        '''

        self._method = 'correlation ' + method

        img = np.asarray(self._mask.copy(), dtype=int)
        regions = measure.regionprops(img)
        bubble = regions[0]

        x0, y0 = bubble.centroid
        r = bubble.major_axis_length / 2.
        if display:
            fign = plt.figure()

        self._shape_fitted = 'circle'
        self._initial_guess = (x0, y0, r)

        def _cost_disk(params):
            x0, y0, r = params
            coords = draw.disk((x0, y0), r, shape=img.shape)
            template = np.zeros_like(img)
            template[coords] = 1
            if display:
                self._dispnobl(template + img, fign)
            return -np.sum((template > 0) & (img > 0))

        res = optimize.minimize(_cost_disk, self._initial_guess,
                                method=method, **keywords)
        self._params = res.x
        self._success = res.success
        if res.success is True:
            return
        elif (method != 'COBYLA' and res.nit == 0):
            raise Exception("Fit circle didn't converge %s" % res)

    def fit_annular_correlation(self,
                                method='Nelder-Mead',
                                display=False,
                                **keywords):
        '''Perform a annular circle fitting on the current mask using
        minimization algorithm  with correlation merit functions.

        Tested with following minimizations methods: 'Nelder-Mead'. Relative
        precision of 1% reached on synthetic images without noise.

        Parameters
        ----------
            method: string, default='Nelder-Mead'
                from scipy.optimize.minimize. Choose among 'Nelder-Mead'
            display: bool, default=False
                SLOWLY shows the progress of the fit.
            **keywords: dict, optional
                passed to scipy.optimize.minimize
        '''

        self._method = 'correlation ' + method
        img = np.asarray(self._mask.copy(), dtype=int)
        regions = measure.regionprops(img)
        bubble = regions[0]

        x0, y0 = bubble.centroid
        r = bubble.major_axis_length / 2.
        inr = r / 2
        if display:
            fign = plt.figure()

        self._shape_fitted = 'annulus'
        self._initial_guess = (x0, y0, r, inr)

        def _cost_annular_disk(params):
            x0, y0, r, inr = params
            coords = draw.disk((x0, y0), r, shape=img.shape)
            template = np.zeros_like(img)
            template[coords] = 1

            coords2 = draw.disk((x0, y0), inr, shape=img.shape)
            template2 = np.zeros_like(img)
            template2[coords2] = 1
            template -= template2

            if display:
                self._dispnobl(template + img, fign)

            merit_fcn = np.sum((template - img)**2)

            return np.sqrt(merit_fcn)

        linear_constraint = LinearConstraint(
            np.identity(4, float32), np.zeros(4),
            np.zeros(4) + np.max(img.shape))
        res = optimize.minimize(_cost_annular_disk, self._initial_guess,
                                method=method, constraints=linear_constraint,
                                **keywords)
        self._params = res.x
        self._success = res.success
        if res.success is False or (method != 'COBYLA' and res.nit == 0):
            raise Exception("Fit circle with hole didn't converge %s" % res)

    def _dispnobl(self, img, fign=None, **kwargs):

        if fign is not None:
            plt.figure(fign.number)
        plt.clf()
        plt.imshow(img, aspect='auto', **kwargs)
        plt.colorbar()
        plt.draw()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.ioff()


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

        tmp = ((x - cc[1]) ** 2 + (y - cc[0]) ** 2) <= r ** 2
        if self._inRadius == 0:
            self._mask = np.where(tmp, False, True)
        else:
            cc = CircularMask(self._shape, self._inRadius, self._maskCenter)
            tmp[cc.asTransmissionValue() > 0] = False
            self._mask = np.where(tmp, False, True)

    @staticmethod
    def fromMaskedArray(maskedArray):
        raise Exception("Not implemented yet. ")
