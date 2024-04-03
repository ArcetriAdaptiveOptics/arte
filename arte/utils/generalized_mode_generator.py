import numpy as np
from scipy.special.basic import factorial
from arte.types.mask import CircularMask
from arte.utils.zernike_generator import ZernikeGenerator
from arte.utils.kl_generator import KLGenerator
from arte.utils.tps_generator import TPSGenerator


class ModeGenerator(object):
    '''
    Generator of modes their derivatives if available ()

    The generator returns masked arrays representing KL polynomials
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
    -- Roddier...


    Examples
    --------

    Create a Zernike polynomial sampled with unit circle defined over 
    64 pixels representing tilt

    >>> zg = ZernikeGenerator(64)
    >>> tip = zg.getZernike(2)
    >>> tilt = zg.getZernike(3)
    >>> tilt = zg[3] # equivalent to getZernike(3) 


    '''

    def __init__(self, pupil, modetype='zernike'):
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
        self._modetype = modetype
        if modetype == 'zernike':
            self._modeGenerator = ZernikeGenerator(pupil)
        elif modetype == 'tps':
            self._modeGenerator = TPSGenerator(pupil)
        elif modetype == 'rbf':
            self._modeGenerator = RBFGenerator(pupil)
        elif modetype == 'kl':
            self._modeGenerator = KLGenerator(pupil)
        else:
            raise ValueError("Invalid mode type %s" % modetype)
    
        

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
        self._modeGenerator._derivativeCoeffX(index)

    def _derivativeCoeffY(self, index):
        self._modeGenerator._derivativeCoeffX(index)

    @classmethod
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

    def getModeCube(self, indexVector):
        ret = np.zeros((len(indexVector), self._shape[0], self._shape[1]))
        for i, index in enumerate(indexVector):
            ret[i] = self.getMode(index)
        return ret

    def getModeDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getKL(index)
        return ret

    def getMode(self, index):
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
        return self._modeGenerator.getMode(index)

    def getDerivativeXDict(self, indexVector):
        return self._modeGenerator.getDerivativeXDict(indexVector)

    def getDerivativeYDict(self, indexVector):
        return self._modeGenerator.getDerivativeYDict(indexVector)

    def getDerivativeX(self, index):
        return self._modeGenerator.getDerivativeX(index)

    def getDerivativeY(self, index):
        return self._modeGenerator.getDerivativeY(index)


