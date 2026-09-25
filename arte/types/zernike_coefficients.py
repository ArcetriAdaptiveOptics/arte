import numpy as np
from arte.types.modal_coefficients import ModalCoefficients
from numbers import Number


class _FirstModeAlias():
    '''
    FIRST_ZERNIKE_MODE descriptor.

    On the class it returns the default first Zernike index (2, i.e. tip:
    piston is not part of the default decomposition). On an instance it is
    an alias of FIRST_MODE, so that reading or setting it at runtime
    changes the labels used by getZ, zernikeIndexes and toDictionary.
    '''

    def __init__(self, default):
        self.default = default

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.default
        return obj.FIRST_MODE

    def __set__(self, obj, value):
        obj.FIRST_MODE = value


class ZernikeCoefficients(ModalCoefficients):
    FIRST_ZERNIKE_MODE = _FirstModeAlias(2)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        value = cls.__dict__.get('FIRST_ZERNIKE_MODE')
        if value is not None and not isinstance(value, _FirstModeAlias):
            cls.FIRST_ZERNIKE_MODE = _FirstModeAlias(value)

    def __init__(self, coefficients, counter=0, first_mode=None):
        if first_mode is None:
            first_mode = type(self).FIRST_ZERNIKE_MODE
        super().__init__(coefficients, counter, first_mode=first_mode)

    def zernikeIndexes(self):
        return self.modeIndexes()

    def getZ(self, zernikeIndexes):
        '''
        Return the coefficient(s) of the given Zernike index(es) (Noll).

        The first stored coefficient is Z_FIRST_ZERNIKE_MODE: by default
        Z2 (tip), since piston is not part of the default decomposition,
        and getZ(1) raises IndexError. See ModalCoefficients.getM.
        '''
        return self.getM(zernikeIndexes)

    @staticmethod
    def fromNumpyArray(coefficientsAsNumpyArray, counter=0, first_mode=None):
        return ZernikeCoefficients(np.array(coefficientsAsNumpyArray), counter,
                                   first_mode=first_mode)

    def _new(self, coefficients):
        return ZernikeCoefficients(coefficients, first_mode=self.FIRST_MODE)

    def __add__(self, other):
        if isinstance(other, ZernikeCoefficients):
            c, first = self._alignedSum(other)
            return ZernikeCoefficients(c, first_mode=first)
        if isinstance(other, Number):
            return self._new(self._coefficients + other)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, ZernikeCoefficients):
            self._coefficients, self.FIRST_MODE = self._alignedSum(other)
            return self
        elif isinstance(other, Number):
            self._coefficients += other
            return self
        return NotImplemented
