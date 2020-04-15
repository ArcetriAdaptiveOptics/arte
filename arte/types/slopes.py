import numpy as np


class Slopes(object):

    def __init__(self, mapX, mapY):
        assert isinstance(mapX, np.ma.MaskedArray)
        assert isinstance(mapY, np.ma.MaskedArray)
        self._mapX = mapX
        self._mapY = mapY
        self._vector = np.hstack((self._mapX.compressed(),
                                 self._mapY.compressed()))
        self._nOfSlopes = self._vector.size

    @staticmethod
    def fromNumpyArray(mapXAsMaskedNumpyArray,
                       mapYAsMaskedNumpyArray):
        return Slopes(mapXAsMaskedNumpyArray,
                      mapYAsMaskedNumpyArray)

    def toNumpyArray(self):
        return self._mapX, self._mapY

#    def pupils(self):
#        return self._pupils
    def numberOfSlopes(self):
        return self._nOfSlopes

    def mapX(self):
        return self._mapX

    def mapY(self):
        return self._mapY

    def vector(self):
        return self._vector

    def vectorX(self):
        return self._vector[0: self._nOfSlopes // 2]

    def vectorY(self):
        return self._vector[self._nOfSlopes // 2:]

    def __eq__(self, o):
        if not isinstance(o, Slopes):
            return False
        if not np.array_equal(self._vector, o._vector):
            return False
        return True

    def __ne__(self, o):
        return not self.__eq__(o)

