import numpy as np
from arte.types.zernike_coefficients import ZernikeCoefficients


class ModalCoefficients(ZernikeCoefficients):
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
        return self.toNumpyArray()[np.array(modeIndexes) - self.FIRST_MODE]

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

