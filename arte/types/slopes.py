import numpy as np


class Slopes(object):
    ''''
    Class that represents one or more generic WFS slope frames
    '''
    def __init__(self, slopesx_compressed, slopesy_compressed, mask2d):
        '''
        Parameters
        ----------
        slopesx_compressed: np.ndarray
            1D slope vector, or 2D slope time series [time, slopes]
        slopesy_compressed: np.ndarray
            1D slope vector, or 2D slope time series [time, slopes]
        mask: np.ndarray
            2D boolean mask to rearrange a 1D slope vector into a 2D frame,
            valid for both X and Y slope vectors.
        '''
        assert slopesx_compressed.shape == slopesy_compressed.shape
        self._mask2d = mask2d
        self._vector = np.hstack((slopesx_compressed, slopesy_compressed))
        if len(self._vector.shape) == 1:
            self._vector = np.expand_dims(self._vector, axis=0)

    @staticmethod
    def from2dmaps(mapX, mapY):
        '''
        Build a Slopes object from X/Y 2D maps or a series of X/Y 2D maps.
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
            are copied into an internal array. No references to the input data are kept.
        '''
        assert isinstance(mapX, np.ma.MaskedArray)
        assert isinstance(mapY, np.ma.MaskedArray)
        assert mapX.shape == mapY.shape
        assert np.all(mapX.mask == mapY.mask)

        mask = mapX.mask.copy()
        if len(mapX.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        nframes = mask.shape[0]
        mask2d = np.logical_or.reduce(mask, axis=0)

        nOfSubaps = np.count_nonzero(~mask2d)
        return Slopes(mapX.compressed().reshape(nframes, nOfSubaps),
                      mapY.compressed().reshape(nframes, nOfSubaps),
                      mask2d)

    @staticmethod
    def fromNumpyArray(mapXAsMaskedNumpyArray,
                       mapYAsMaskedNumpyArray):
        return Slopes.from2dmaps(mapXAsMaskedNumpyArray,
                                 mapYAsMaskedNumpyArray)

    def toNumpyArray(self):
        return self.mapX(), self.mapY()

    def numberOfSlopes(self):
        return self._vector.shape[1]

    def numberOfFrames(self):
        return len(self._vector)

    @property
    def shape2d(self):
        return self._mask2d.shape

    def _map2d(self, data1d):
        fullshape = (self.numberOfFrames(),) + self.shape2d
        data2d = np.zeros_like(self._vector, shape=fullshape)
        data2d[:, *np.where(~self._mask2d)] = data1d
        return np.ma.array(data2d, mask=np.broadcast_to(self._mask2d, shape=fullshape))

    def mapX(self, add_time_axis=False):
        mapX = self._map2d(self._vector[:, :self.numberOfSlopes()//2])
        if not add_time_axis:
            return np.squeeze(mapX)
        else:
            return mapX

    def mapY(self, add_time_axis=False):
        mapY = self._map2d(self._vector[:, self.numberOfSlopes()//2:])
        if not add_time_axis:
            return np.squeeze(mapY)
        else:
            return mapY

    def vector(self):
        return np.squeeze(self._vector)

    def vectorX(self):
        return np.squeeze(self._vector[:, 0:self.numberOfSlopes() // 2])

    def vectorY(self):
        return np.squeeze(self._vector[:, self.numberOfSlopes() // 2:])

    def __eq__(self, o):
        if not isinstance(o, Slopes):
            return False
        return np.array_equal(self._vector, o._vector)

    def __ne__(self, o):
        return not self.__eq__(o)

