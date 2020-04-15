import numpy as np


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
