import numpy as np


class Slopes(object):
    ''''
    Class that represents a generic WFS slope frame
    '''
    def __init__(self, slopesx_compressed, slopesy_compressed, mask2d):
        '''
        Parameters
        ----------
        slopesx_compressed: np.ndarray
            1D slope vector
        slopesy_compressed: np.ndarray
            1D slope vector
        mask: np.ndarray
            2D boolean mask to rearrange a 1D slope vector into a 2D frame,
            valid for both X and Y slope vectors.
        '''
        assert len(slopesx_compressed) == len(slopesy_compressed), \
            'X and Y slopes length differ'
        self._mask2d = mask2d
        self._vector = np.hstack((slopesx_compressed, slopesy_compressed))

    @staticmethod
    def from_2dmaps(mapX, mapY):
        '''
        Build a Slopes object from X/Y 2D maps or a series of X/Y 2D maps.
        Both maps must be np.ma.MaskedArray objects. Masks for X and Y slopes
        must be identical.

        Parameters
        ----------
        mapX: np.ma.MaskedArray
            2D slope map
        mapY: np.ma.MaskedArray
            2D slope map

        Returns
        -------
        slopes: Slopes object
            A new Slopes object. Valid slopes (as indicated by the mask)
            are copied into an internal array. No references to the input data are kept.
        '''
        assert isinstance(mapX, np.ma.MaskedArray), 'mapX is not a np.ma.MaskedArray'
        assert isinstance(mapY, np.ma.MaskedArray), 'mapY is not a np.ma.MaskedArray'
        assert mapX.shape == mapY.shape, 'mapX and mapY shapes differ'
        assert np.all(mapX.mask == mapY.mask), 'mapX and mapY masks differ'
        return Slopes(mapX.compressed(), mapY.compressed(), np.ma.getmaskarray(mapX))

    @staticmethod
    def fromNumpyArray(mapXAsMaskedNumpyArray,
                       mapYAsMaskedNumpyArray):
        return Slopes.from_2dmaps(mapXAsMaskedNumpyArray,
                                  mapYAsMaskedNumpyArray)

    def toNumpyArray(self):
        return self.mapX(), self.mapY()

    def numberOfSlopes(self):
        return len(self._vector)

    @property
    def shape2d(self):
        return self._mask2d.shape

    def _map2d(self, data1d):
        data2d = np.zeros_like(self._vector, shape=self._mask2d.shape)
        data2d[np.where(~self._mask2d)] = data1d
        return np.ma.array(data2d, mask=self._mask2d)

    def mapX(self):
        return self._map2d(self.vectorX())

    def mapY(self):
        return self._map2d(self.vectorY())

    def vector(self):
        return self._vector

    def vectorX(self):
        return self._vector[:self.numberOfSlopes() // 2]

    def vectorY(self):
        return self._vector[self.numberOfSlopes() // 2:]

    def __eq__(self, o):
        if not isinstance(o, Slopes):
            return False
        return np.array_equal(self._vector, o._vector)

    def __ne__(self, o):
        return not self.__eq__(o)

