import numpy as np
from arte.types.modal_coefficients import ModalCoefficients


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
