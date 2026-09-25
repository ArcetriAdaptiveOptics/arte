import numpy as np
from numbers import Number


class ModalCoefficients():
    FIRST_MODE = 0

    def __init__(self, coefficients, counter=0, first_mode=0):
        self._coefficients = coefficients
        self._counter = counter
        self.FIRST_MODE = first_mode

    def modeIndexes(self):
        return np.arange(self.FIRST_MODE, self.FIRST_MODE + self.numberOfModes())

    def numberOfModes(self):
        return len(self._coefficients)

    def getM(self, modeIndexes):
        '''
        Return the coefficient(s) of the given mode index(es).

        Mode indexes are counted from FIRST_MODE: getM(FIRST_MODE) is the
        first element of toNumpyArray().

        Parameters
        ----------
        modeIndexes: int or sequence of int
            Mode index(es) in the range
            [FIRST_MODE, FIRST_MODE + numberOfModes() - 1].

        Raises
        ------
        IndexError
            If any index is outside that range, or if boolean indexes
            are given.
        '''
        idx = np.array(modeIndexes)
        if idx.dtype.kind == 'b':
            raise IndexError(
                '%s: boolean mode indexes are not supported' %
                self.__class__.__name__)
        idx = idx - self.FIRST_MODE
        if idx.dtype.kind in 'iu' and idx.size > 0 and (
                idx < 0 if idx.ndim == 0 else idx.min() < 0):
            raise IndexError(
                '%s: mode index(es) %s smaller than FIRST_MODE=%d' % (
                    self.__class__.__name__,
                    np.atleast_1d(modeIndexes)[np.atleast_1d(idx) < 0],
                    self.FIRST_MODE))
        return self.toNumpyArray()[idx]

    def toDictionary(self):
        keys = self.modeIndexes()
        values = self._coefficients
        return dict(list(zip(keys, values)))

    def toNumpyArray(self):
        return self._coefficients

    @staticmethod
    def fromNumpyArray(coefficientsAsNumpyArray, counter=0, **kwargs):
        return ModalCoefficients(np.array(coefficientsAsNumpyArray), counter, **kwargs)

    def counter(self):
        return self._counter

    def setCounter(self, counter):
        self._counter = counter

    def _new(self, coefficients):
        '''
        Return a new instance holding `coefficients`, with the same mode
        labels (FIRST_MODE) as self. Used by the arithmetic operators.
        '''
        return ModalCoefficients(coefficients, first_mode=self.FIRST_MODE)

    def _alignedSum(self, other):
        '''
        Sum coefficients mode by mode. Returns (coefficients, first_mode).

        When both operands have the same FIRST_MODE the result is the
        historical one (shorter array added to the head of the longer).
        Otherwise the arrays are aligned on the mode index and the
        result spans the union of the two mode ranges, missing modes
        counting as zero.
        '''
        a = self._coefficients
        b = other._coefficients
        if self.FIRST_MODE == other.FIRST_MODE:
            if len(a) < len(b):
                c = b.copy()
                c[:len(a)] += a
            else:
                c = a.copy()
                c[:len(b)] += b
            return c, self.FIRST_MODE
        first = min(self.FIRST_MODE, other.FIRST_MODE)
        last = max(self.FIRST_MODE + len(a), other.FIRST_MODE + len(b))
        c = np.zeros(last - first, dtype=np.result_type(a, b))
        c[self.FIRST_MODE - first:self.FIRST_MODE - first + len(a)] += a
        c[other.FIRST_MODE - first:other.FIRST_MODE - first + len(b)] += b
        return c, first

    def __eq__(self, o):
        if self._counter != o._counter:
            return False
        if self.FIRST_MODE != o.FIRST_MODE:
            return False
        if not np.array_equal(self._coefficients, o._coefficients):
            return False
        return True

    def __ne__(self, o):
        return not self.__eq__(o)

    def __str__(self):
        return str(self._coefficients)

    def __add__(self, other):
        if isinstance(other, ModalCoefficients):
            c, first = self._alignedSum(other)
            return ModalCoefficients(c, first_mode=first)
        if isinstance(other, Number):
            return self._new(self._coefficients + other)
        return NotImplemented

    def __radd__(self, other):
      return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, ModalCoefficients):
            self._coefficients, self.FIRST_MODE = self._alignedSum(other)
            return self
        elif isinstance(other, Number):
            self._coefficients += other
            return self
        return NotImplemented

    def __neg__(self):
        return self._new(-self._coefficients)

    def __pos__(self):
        return self._new(self._coefficients)

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
            return self._new(self._coefficients * other)
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
            return self._new(self._coefficients / other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return self._new(other / self._coefficients)
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, Number):
            self._coefficients /= other
            return self
        return NotImplemented
