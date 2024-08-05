import numpy as np
from numbers import Number

class ZernikeCoefficients(object):
    FIRST_ZERNIKE_MODE = 2

    def __init__(self, coefficients, counter=0):
        self._coefficients = coefficients
        self._counter = counter

    def zernikeIndexes(self):
        return np.arange(self.FIRST_ZERNIKE_MODE,
                         self.FIRST_ZERNIKE_MODE + self.numberOfModes())

    def numberOfModes(self):
        return len(self._coefficients)

    def getZ(self, zernikeIndexes):
        return self.toNumpyArray()[np.array(zernikeIndexes) - 
                                   self.FIRST_ZERNIKE_MODE]

    def toDictionary(self):
        keys = self.zernikeIndexes()
        values = self._coefficients
        return dict(list(zip(keys, values)))

    def toNumpyArray(self):
        return self._coefficients

    @staticmethod
    def fromNumpyArray(coefficientsAsNumpyArray, counter=0):
        return ZernikeCoefficients(np.array(coefficientsAsNumpyArray), counter)

    def counter(self):
        return self._counter

    def setCounter(self, counter):
        self._counter = counter

    def __eq__(self, o):
        if self._counter != o._counter:
            return False
        if not np.array_equal(self._coefficients, o._coefficients):
            return False
        return True

    def __ne__(self, o):
        return not self.__eq__(o)

    def __str__(self):
        return str(self._coefficients)

    def __add__(self, other):
        if isinstance(other, ZernikeCoefficients):
            if len(self._coefficients) < len(other._coefficients):
                c = other._coefficients.copy()
                c[:len(self._coefficients)] += self._coefficients
            else:
                c = self._coefficients.copy()
                c[:len(other._coefficients)] += other._coefficients
            return ZernikeCoefficients(c)
        if isinstance(other, Number):
            return ZernikeCoefficients(self._coefficients + other)
        return NotImplemented

    def __radd__(self, other):
      return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, ZernikeCoefficients):
            if len(self._coefficients) < len(other._coefficients):
                c = other._coefficients.copy()
                c[:len(self._coefficients)] += self._coefficients
            else:
                c = self._coefficients.copy()
                c[:len(other._coefficients)] += other._coefficients
            self._coefficients = c
            return self
        elif isinstance(other, Number):
            self._coefficients += other
            return self
        return NotImplemented

    def __neg__(self):
        return ZernikeCoefficients(-self._coefficients)

    def __pos__(self):
        return ZernikeCoefficients(self._coefficients)

    def __abs__(self):
        return pow(sum(coo**2 for coo in self._coefficients), 0.5)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        self += -other
        return self

    def __mul__(self, other):
        if isinstance(other, Number):
            return ZernikeCoefficients(self._coefficients * other)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if isinstance(other, Number):
            self._coefficients *= other
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return ZernikeCoefficients(self._coefficients / other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return ZernikeCoefficients(other / self._coefficients)
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, Number):
            self._coefficients /= other
            return self
        return NotImplemented
