import numpy as np
from arte.types.modal_coefficients import ModalCoefficients
from numbers import Number


class ZernikeCoefficients(ModalCoefficients):
    FIRST_ZERNIKE_MODE = 2

    def __init__(self, coefficients, counter=0):
        super().__init__(coefficients, counter, first_mode=self.FIRST_ZERNIKE_MODE)

    def zernikeIndexes(self):
        return self.modeIndexes()

    def getZ(self, zernikeIndexes):
        return self.getM(zernikeIndexes)

    @staticmethod
    def fromNumpyArray(coefficientsAsNumpyArray, counter=0):
        return ZernikeCoefficients(np.array(coefficientsAsNumpyArray), counter)

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
