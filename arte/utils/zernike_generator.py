import numpy as np
from scipy.special.basic import factorial
from arte.types.mask import CircularMask


class ZernikeGenerator(object):
    '''
    Generator of Zernike polynomials and their derivatives

    The generator returns masked arrays representing Zernike polynomials
    sampled over the array grid. The pixels whose distance from the unit disk
    center is greater than the mask radius are masked.

    The discretization of the continous unit disk space into the array grid is
    done in the pixels' center (e.g. if the disk diameter is 2 pixels and the
    center is in x=1 then the first pixel contains polynomial values
    evaluated in x=-0.5 and the second pixel contains values evaluated in x=0.5)

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


    Notes
    -----
    Polynomials normalization and ordering follows the conventions
    described in Noll's paper [1]_

    .. [1] Noll, R. J., “Zernike polynomials and atmospheric
       turbulence.”, Journal of the Optical Society of America
       (1917-1983), vol. 66, pp. 207–211, 1976.


    Examples
    --------

    Create a Zernike polynomial sampled with unit circle defined over 
    64 pixels representing tilt

    >>> zg = ZernikeGenerator(64)
    >>> tip = zg.getZernike(2)
    >>> tilt = zg.getZernike(3)
    >>> tilt = zg[3] # equivalent to getZernike(3) 


    '''

    def __init__(self, pupil):
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
                self._shape, maskCenter=self._center, maskRadius=self._radius)
            self._boolean_mask = cm.mask()

        self._rhoMap, self._thetaMap = self._polar_array()
        self._dx = None
        self._dy = None
        self._dictCache = {}
        self._dictDxCache = {}
        self._dictDyCache = {}

    def radius(self):
        '''
        Radius of the unit disk 

        Returns
        -------
        radius: real
            unit disk radius in pixel

        '''
        return self._radius

    def center(self):
        '''
        Y, X coordinates of the unit disk center

        Returns
        -------
        center: `~numpy.array` of shape (2,)
            Y, X coordinate of the unit disk center in the array reference system

        '''
        return self._center

    def _derivativeCoeffX(self, index):
        if (self._dx is None) or (self._dx.shape[0] < index):
            self._dx = self._computeDerivativeCoeffX(index)
        return self._dx[0:index, 0:index]

    def _derivativeCoeffY(self, index):
        if (self._dy is None) or (self._dy.shape[0] < index):
            self._dy = self._computeDerivativeCoeffY(index)
        return self._dy[0:index, 0:index]

    @classmethod
    def degree(cls, index):
        n = cls.radial_order(index)
        cn = n * (n + 1) / 2 + 1
        if n % 2 == 0:
            m = int(index - cn + 1) // 2 * 2
        else:
            m = int(index - cn) // 2 * 2 + 1
        radialDegree = n
        azimuthalFrequency = m
        return radialDegree, azimuthalFrequency

    def _rnm(self, radialDegree, azimuthalFrequency, rhoArray):
        n = radialDegree
        m = azimuthalFrequency
        rho = rhoArray
        if (n - m) % 2 != 0:
            raise Exception("n-m must be even. Got %d-%d" % (n, m))
        if abs(m) > n:
            raise Exception("The following must be true |m|<=n. Got %d, %d" %
                            (n, m))

        if(n == 0 and m == 0):
            return np.ones(rho.shape)
        rho = np.where(rho < 0, 0, rho)
        Rnm = np.zeros(rho.shape)
        S = (n - abs(m)) // 2
        for s in range(0, S + 1):
            CR = pow(-1, s) * factorial(n - s) / \
                (factorial(s) * factorial(-s + (n + abs(m)) / 2) *
                 factorial(-s + (n - abs(m)) / 2))
            p = CR * pow(rho, n - 2 * s)
            Rnm = Rnm + p
        return Rnm

    def _polar(self, index, rhoArray, thetaArray):
        n, m = self.degree(index)
        rho = rhoArray
        theta = thetaArray

        Rnm = self._rnm(n, m, rho)
        NC = np.sqrt(2 * (n + 1))
        if m == 0:
            return np.sqrt(0.5) * NC * Rnm
        if index % 2 == 0:
            return NC * Rnm * np.cos(m * theta)
        else:
            return NC * Rnm * np.sin(m * theta)

    def cartesian_coordinates(self):
        '''
        Return X, Y maps of cartesian coordinates

        The Zernike polynomials and derivatives are evaluated over a grid of 
        points, whose coordinates are accessible through `cartesian_coordinates`

        Returns
        -------
        X,Y: `~numpy.array`
            coordinates of points of the unit disk where the polynomials
            are evaluated

        Examples
        --------

        Create Zernike polynomials on a unit disk defined over 
        4 pixels. The returned arrays have (4,4) shape (corresponding 
        to the coords [-0.75, -0.25, 0.25, 0.75])

        >>> zg = ZernikeGenerator(4)
        >>> x, y = zg.cartesian_coordinates()
        >>> x[0]
        array([-0.75, -0.25,  0.25,  0.75])

        As above with 3 pixels

        >>> zg = ZernikeGenerator(3)
        >>> x, y = zg.cartesian_coordinates()
        >>> x[0]
        array([-0.666, 0,  0.666])

        In case of non-integer diameter, the array size is rounded up to the
        next integer

        >>> zg = ZernikeGenerator(2.5)
        >>> x, y = zg.cartesian_coordinates()
        >>> x[0]
        array([-0.8, 0,  0.8])

        '''
        nPxY = self._shape[0]
        nPxX = self._shape[1]
        c = np.array(self.center())
        cc = np.expand_dims(c, axis=(1, 2))
        Y, X = (np.mgrid[0.5: nPxY + 0.5: 1,
                         0.5: nPxX + 0.5: 1] - cc) / self.radius()

        return X, Y

    def _polar_array(self):
        X, Y = self.cartesian_coordinates()
        r = np.sqrt(X ** 2 + Y ** 2)
        th = np.arctan2(Y, X)
        return r, th

    def getZernikeDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getZernike(index)
        return ret

    def getZernike(self, index):
        if not self._is_integer_num(index):
            raise ValueError("Invalid Zernike index %s" % index)
        if index <= 0:
            raise ValueError("Invalid Zernike index %d" % index)
        if index not in list(self._dictCache.keys()):
            res = self._polar(index, self._rhoMap,
                              self._thetaMap)
            self._dictCache[index] = np.ma.masked_array(
                data=res, mask=self._boolean_mask)
        return self._dictCache[index]

    @staticmethod
    def _is_integer_num(n):
        if isinstance(n, (int, np.integer)):
            return True
        if isinstance(n, float):
            return n.is_integer()
        return False

    def __getitem__(self, index):
        return self.getZernike(index)

    def _computeDerivativeCoeffX(self, index):
        jmax = index
        G_mat = np.zeros((jmax, jmax))
        for i in range(1, jmax + 1):
            for j in range(1, jmax + 1):
                ni, mi = ZernikeGenerator.degree(i)
                nj, mj = ZernikeGenerator.degree(j)
                if (
                    (
                        (
                            (
                                mi != 0 and mj != 0
                            ) and (
                                (
                                    _isEven(i) and _isEven(j)
                                ) or (
                                    _isOdd(i) and _isOdd(j)
                                )
                            )
                        ) or (
                            (
                                (mi == 0) and _isEven(j)
                            ) or (
                                (mj == 0) and _isEven(i)
                            )
                        )
                    ) and (
                        (mj == mi + 1) or (mj == mi - 1)
                    ) and (
                        j < i
                    )
                ):
                    G_mat[i - 1, j - 1] = np.sqrt((ni + 1) * (nj + 1))
                    if ((mi == 0) or (mj == 0)):
                        G_mat[i - 1, j - 1] *= np.sqrt(2)
        return G_mat

    def _computeDerivativeCoeffY(self, index):
        jmax = index
        G_mat = np.zeros((jmax, jmax))
        for i in range(1, jmax + 1):
            for j in range(1, jmax + 1):
                ni, mi = ZernikeGenerator.degree(i)
                nj, mj = ZernikeGenerator.degree(j)
                if (
                    (
                        (
                            (
                                mi != 0 and mj != 0
                            ) and (
                                (
                                    _isOdd(i) and _isEven(j)
                                ) or (
                                    _isEven(i) and _isOdd(j)
                                )
                            )
                        ) or (
                            (
                                (mi == 0) and _isOdd(j)
                            ) or (
                                (mj == 0) and _isOdd(i)
                            )
                        )
                    ) and (
                        (mj == mi + 1) or (mj == mi - 1)
                    ) and (
                        j < i
                    )
                ):
                    G_mat[i - 1, j - 1] = np.sqrt((ni + 1) * (nj + 1))
                    if ((mi == 0) or (mj == 0)):
                        G_mat[i - 1, j - 1] *= np.sqrt(2)
                    if (
                        (
                            (
                                (mj == mi + 1) and _isOdd(i)
                            ) or (
                                (mj == mi - 1) and _isEven(i)
                            )
                        ) and (
                            mi != 0
                        )
                    ):
                        G_mat[i - 1, j - 1] *= -1

        return G_mat

    def getDerivativeXDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getDerivativeX(index)
        return ret

    def getDerivativeYDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getDerivativeY(index)
        return ret

    def getDerivativeX(self, index):
        if index not in self._dictDxCache:
            self._dictDxCache[index] = self._computeDerivativeX(index)
        return self._dictDxCache[index]

    def _computeDerivativeX(self, index):
        coeffX = self._derivativeCoeffX(index)
        dx = self.getZernike(1) * 0.
        for i in range(1, index):
            dx += coeffX[index - 1, i - 1] * self.getZernike(i)
        return dx

    def getDerivativeY(self, index):
        if index not in self._dictDyCache:
            self._dictDyCache[index] = self._computeDerivativeY(index)
        return self._dictDyCache[index]

    def _computeDerivativeY(self, index):
        coeffY = self._derivativeCoeffY(index)
        dy = self.getZernike(1) * 0.
        for i in range(1, index):
            dy += coeffY[index - 1, i - 1] * self.getZernike(i)
        return dy

    @classmethod
    def radial_order(cls, j):
        '''
        Return radial order of j-th polynomial

        Parameters
        ----------
        j: int or sequence of int
            polynomial index

        Returns
        -------
        n: int or sequence of int
            radial order of the specified indexes

        '''
        return np.ceil(0.5 * (np.sqrt(8 * np.array(j) + 1) - 3)).astype(int)


def _isOdd(num):
    return num % 2 != 0


def _isEven(num):
    return num % 2 == 0
