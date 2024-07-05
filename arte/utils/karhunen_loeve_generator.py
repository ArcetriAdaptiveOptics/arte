from arte.types.mask import CircularMask


import numpy as np


class KarhunenLoeveGenerator(object):
    """
    Generator of KL polynomials departing from a covariance matrix and a user
    defined base of polynomials.

    The user can specify the diameter of the pupil `pupil` or
    specify a `~arte.types.mask.CircularMask`

    In the case a scalar real value is passed as `pupil` argument,
    The shape of the returned array cannot be specified: it is a
    square array of size ceil(pupil_diameter).
    The center of the unit disk cannot be specified and it is the central pixel
    of the array (if the array size is odd) or the corner of the
    4 central pixels (if the array size is even).


    In the case a `~arte.types.mask.CircularMask` is passed as `pupil` argument,
    the mask is used as unit disk over which the polynomial are computed. Non-integer
    center coordinates and radius are properly managed.



    Parameters
    ----------
        pupil: real or `~arte.types.mask.CircularMask`
            If a scalar value, the argument is used as pupil diameter in pixels.
            If a `~arte.types.mask.CircularMask`, the argument is used as mask
            representing the unit disk.
        covariance_matrix:  `~numpy.array` of shape (nModes, nModes) representing
                            the phenomenon covariance matrix


    Examples
    --------

    Create a KL polynomial using Zernike description of Kolmogorov turbulence
    >>> from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
    >>> from arte.utils.zernike_generator import ZernikeGenerator
    >>> from arte.utils.kl_generator import KLGenerator
    >>> from arte.types.mask import CircularMask
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> mask = CircularMask((128,128), 64, (64,64))
    >>> zz = ZernikeGenerator(mask)
    >>> nmodes=100
    >>> kolm_covar = getFullKolmogorovCovarianceMatrix(nmodes)
    >>> zbase = np.rollaxis(np.ma.masked_array([zz.getZernike(n)
                                            for n in range(2,nmodes+2)]),0,3)
    >>> generator = KLGenerator(mask, kolm_covar)
    >>> generator.generateFromBase(zbase)
    >>> kl =generator.getKL(2)
    >>> plt.imshow(kl)


    """

    def __init__(self, pupil, covariance_matrix):
        if isinstance(pupil, CircularMask):
            self._radius = pupil.radius()
            self._shape = pupil.shape()
            self._center = pupil.center()
            self._boolean_mask = pupil.mask()
        else:
            self._radius = pupil / 2
            sz = np.ceil(pupil)
            self._shape = (sz, sz)
            self._center = np.ones(2) * (sz / 2)
            cm = CircularMask(
                self._shape, maskCenter=self._center, maskRadius=self._radius
            )
            self._boolean_mask = cm.mask()

        self._covariance_matrix = covariance_matrix
        self._rhoMap, self._thetaMap = self._polar_array()
        self._dx = None
        self._dy = None
        self._dictCache = {}
        self._dictDxCache = {}
        self._dictDyCache = {}
        self._nmodes = covariance_matrix.shape[0]
        self._inpupil = pupil
        self._klBase = None

    def radius(self):
        """
        Radius of the unit disk

        Returns
        -------
        radius: real
            unit disk radius in pixel

        """
        return self._radius

    def center(self):
        """
        Y, X coordinates of the unit disk center

        Returns
        -------
        center: `~numpy.array` of shape (2,)
            Y, X coordinate of the unit disk center in the array reference system

        """
        return self._center

    def first_mode(self):
        '''
        First mode index
        '''
        return 0
    
    def _derivativeCoeffX(self, index):
        raise Exception("Not available here")

    def _derivativeCoeffY(self, index):
        raise Exception("Not available here")

    @classmethod
    def degree(cls, index):
        raise Exception("Not available here")

    def _rnm(self, radialDegree, azimuthalFrequency, rhoArray):
        raise Exception("Not available here")

    def _polar(self, index, rhoArray, thetaArray):
        raise Exception("Not available here")

    def cartesian_coordinates(self):
        """
        Return X, Y maps of cartesian coordinates

        The KL polynomials and derivatives are evaluated over a grid of
        points, whose coordinates are accessible through `cartesian_coordinates`

        Returns
        -------
        X,Y: `~numpy.array`
            coordinates of points of the unit disk where the polynomials
            are evaluated

        Examples
        --------

        Create KL polynomials on a unit disk defined over
        4 pixels. The returned arrays have (4,4) shape (corresponding
        to the coords [-0.75, -0.25, 0.25, 0.75])

        >>> zg = KLGenerator(4)
        >>> x, y = zg.cartesian_coordinates()
        >>> x[0]
        array([-0.75, -0.25,  0.25,  0.75])

        As above with 3 pixels

        >>> zg = KLGenerator(3)
        >>> x, y = zg.cartesian_coordinates()
        >>> x[0]
        array([-0.666, 0,  0.666])

        In case of non-integer diameter, the array size is rounded up to the
        next integer

        >>> zg = KLGenerator(2.5)
        >>> x, y = zg.cartesian_coordinates()
        >>> x[0]
        array([-0.8, 0,  0.8])

        """
        nPxY = self._shape[0]
        nPxX = self._shape[1]
        c = np.array(self.center())
        cc = np.expand_dims(c, axis=(1, 2))
        Y, X = (
            np.mgrid[0.5 : nPxY + 0.5 : 1, 0.5 : nPxX + 0.5 : 1] - cc
        ) / self.radius()

        return X, Y

    def _polar_array(self):
        X, Y = self.cartesian_coordinates()
        r = np.sqrt(X**2 + Y**2)
        th = np.arctan2(Y, X)
        return r, th

    def generateFromBase(self, inBase):
        # Generate the KL base from a base of polynomials.
        # The input base is a cube of polynomials in the form (nPx, nPx, nmodes)
        # where nPx is the size of the square array where the polynomials are defined
        # and nmodes is the number of modes in the base.
        # The base is used to compute the Karhunen-Loeve modes of the phenomena
        # defined by the covariance matrix.
        # The base is internally stored and can be accessed through the getCube
        # method.
        # The base is computed as the dot product of the input base and the
        # eigenvectors of the covariance matrix.
        # The input base is used to compute the Karhunen-Loeve modes, which are
        # stored in the _klBase attribute.

        shapex, shapey, nmodes = inBase.shape
        if nmodes != self._nmodes:
            raise ValueError(
                "The number of modes in the base is different from the number of modes in the covariance matrix"
            )

        self._inBase = inBase
        # zcube_masked = zcube.compressed().reshape(np.size(zcube.compressed())//nzern,nzern)
        # Compute the eigenvalues and eigenvectors
        U, S, Vt = np.linalg.svd(self._covariance_matrix)

        # The eigenvectors are the Karhunen-Loeve modes
        klcube = inBase.dot(Vt.T)

        # Remove piston from each mode
        klcube = klcube - np.mean(klcube, axis=(0, 1))
        self._klBase = klcube
        # refill the _dictCache with the new base
        self._dictCache = {i: klcube[:, :, i] for i in range(0, nmodes)}
        self._klBase = np.rollaxis(
            np.ma.masked_array([self.getKL(n) for n in range(self._nmodes)]), 0, 3
        )

    def getCube(self):
        # Return the base of KL polynomials
        if self._klBase is None:
            raise ValueError("KL base not generated")
        else:
            return self._klBase

    def getKLDict(self, indexVector):
        # Return a dictionary of KL polynomials indexed by the input vector
        ret = {}
        for index in indexVector:
            ret[index] = self.getKL(index)
        return ret

    def getModesDict(self, indexVector):
        return self.getKLDict(indexVector)

    def getKL(self, index):
        # Return the KL polynomial corresponding to the input index
        if not self._is_integer_num(index):
            raise ValueError("Invalid KL index %s not integer" % index)
        if index < 0:
            raise ValueError("Invalid KL index %d less than 0" % index)
        if index not in list(self._dictCache.keys()):
            raise ValueError("Invalid KL index %d, out of base" % index)
        return self._dictCache[index]

    @staticmethod
    def _is_integer_num(n):
        if isinstance(n, (int, np.integer)):
            return True
        if isinstance(n, float):
            return n.is_integer()
        return False

    def __getitem__(self, index):
        return self.getKL(index)

    def getDerivativeXDict(self, indexVector):
        raise Exception("Not implemented yet")

    def getDerivativeYDict(self, indexVector):
        raise Exception("Not implemented yet")

    def getDerivativeX(self, index):
        raise Exception("Not implemented yet")
        # if index not in self._dictDxCache:
        #    self._dictDxCache[index] = self._computeDerivativeX(index)
        # return self._dictDxCache[index]

    def _computeDerivativeX(self, index):
        pass

    def getDerivativeY(self, index):
        raise Exception("Not implemented yet")

    def _computeDerivativeY(self, index):
        pass
