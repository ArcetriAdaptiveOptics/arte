import numpy as np


class Slopes(object):
    ''''
    Class that represents one or more generic WFS slope frames
    '''
    def __init__(self, slopesx_compressed, slopesy_compressed, mask):
        '''
        Parameters
        ----------
        slopesx_compressed: np.ndarray
            1D or 2D slope vector. If 2D, it is assumed to be [time, slopes]
        slopesy_compressed: np.ndarray
            1D or 2D slope vector. If 2D, it is assumed to be [time, slopes]
        mask: np.ndarray
            2D boolean mask, valid for both X and Y slope vectors
        '''
        self._mask = mask
        self._vector = np.hstack((slopesx_compressed, slopesy_compressed))
        if len(self._vector.shape) == 1:
            self._vector = np.expand_dims(self._vector, axis=0)

    @staticmethod
    def from2dmaps(mapX, mapY):
        '''
        Build a Slopes object from X/Y 2d maps or a series of X/Y 2d maps.
        Both maps must be np.ma.MaskedArray objects. Masks for X and Y slopes
        must be identical.

        Parameters
        ----------
        mapX: np.ma.MaskedArray
            2D or 3D slope map. If 3D, it is assumed to be [time, rows, cols]
        mapY: np.ma.MaskedArray
            2D or 3D slope map. If 3D, it is assumed to be [time, rows, cols]

        Returns
        -------
        slopes: Slopes object
            A new Slopes object. Valid slopes (as indicated by the mask)
            is copied into an internal array. No references to the input data are kept.
        '''
        assert isinstance(mapX, np.ma.MaskedArray)
        assert isinstance(mapY, np.ma.MaskedArray)
        assert mapX.shape == mapY.shape
        assert np.all(mapX.mask == mapY.mask)

        mask = mapX.mask.copy()
        if len(mapX.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        nframes = mapX.shape[0]
        single_mask = np.logical_or.reduce(mask, axis=0)

        nOfSubaps = np.count_nonzero(~single_mask)
        vector = np.hstack((mapX.compressed().reshape(nframes, nOfSubaps),
                            mapY.compressed().reshape(nframes, nOfSubaps)))
        return Slopes(vector, single_mask)

    @staticmethod
    def fromNumpyArray(mapXAsMaskedNumpyArray,
                       mapYAsMaskedNumpyArray):
        return Slopes(mapXAsMaskedNumpyArray,
                      mapYAsMaskedNumpyArray)

    def toNumpyArray(self):
        return self.mapX(), self.mapY()

    def numberOfSlopes(self):
        return self._vector.shape[1]

    def numberOfFrames(self):
        return len(self._vector)

    @property
    def shape2d(self):
        return self._mask.shape

    def mapX(self):
        fullshape = (self.numberOfFrames(),) + self.shape2d
        data = np.zeros_like(self._vector, shape=fullshape)
        data[:, *np.where(~self._mask)] = self._vector[:, :self.numberOfSlopes()//2]
        return np.squeeze(np.ma.array(data, mask=np.broadcast_to(self._mask, shape=fullshape)))

    def mapY(self):
        fullshape = (self.numberOfFrames(),) + self.shape2d
        data = np.zeros_like(self._vector, shape=fullshape)
        data[:, *np.where(~self._mask)] = self._vector[:, self.numberOfSlopes()//2:]
        return np.squeeze(np.ma.array(data, mask=np.broadcast_to(self._mask, shape=fullshape)))

    def vector(self):
        return np.squeeze(self._vector)

    def vectorX(self):
        return np.squeeze(self._vector[:, 0: self.numberOfSlopes() // 2])

    def vectorY(self):
        return np.squeeze(self._vector[:, self.numberOfSlopes() // 2:])

    def __eq__(self, o):
        if not isinstance(o, Slopes):
            return False
        if not np.array_equal(self._vector, o._vector):
            return False
        return True

    def __ne__(self, o):
        return not self.__eq__(o)

