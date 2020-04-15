import numpy as np
from arte.utils.radial_profile import computeRadialProfile
from scipy import interpolate


__version__= "$Id: $"


class ScalarBidimensionalFunction(object):
    '''
    Represents a scalar function in an XY plane
    '''

    def __init__(self, valuesArray, xCoordOfValuesArray, yCoordOfValuesArray):
        self._values= valuesArray
        self._xCoord= xCoordOfValuesArray
        self._yCoord= yCoordOfValuesArray
        self._checkPassedArrays()
        self._step= self._computeStep()
        self._xOriginInPx= self._computeXOriginInPx()
        self._yOriginInPx= self._computeYOriginInPx()


    def _checkPassedArrays(self):
        self._checkShapes()
        self._checkXCoordRawsAreConstants()
        self._checkYCoordColumnsAreConstants()


    def _checkShapes(self):
        assert self.values().shape == self.xCoord().shape
        assert self.yCoord().shape == self.xCoord().shape


    def _checkXCoordRawsAreConstants(self):
        assert not np.any(np.ptp(self.xCoord(), axis=0))


    def _checkYCoordColumnsAreConstants(self):
        assert not np.any(np.ptp(self.yCoord(), axis=1))


    def _computeStep(self):
        xStep= self._computeXStep()
        #assert xStep == self._computeYStep()
        return xStep


    def _computeXStep(self):
        arr= self.xCoord()[0, :]
        return self._computeStepOfUniformlySpacedVector(arr)


    def _computeYStep(self):
        arr= self.yCoord()[:, 0]
        return self._computeStepOfUniformlySpacedVector(arr)


    def _computeStepOfUniformlySpacedVector(self, vector):
        delta= vector[1:] - vector[:-1]
        assert np.isclose(0, delta.ptp(), atol=1e-6 * delta[0])
        return delta[0]


    def _computeXOriginInPx(self):
        arr= self.xCoord()[0, :]
        return self._interpolateForValue(arr, 0.0)


    def _computeYOriginInPx(self):
        arr= self.yCoord()[:, 0]
        return self._interpolateForValue(arr, 0.0)


    def _interpolateForValue(self, arr, value):
        arrSz= arr.shape[0]
        return np.interp(value, arr, np.arange(arrSz),
                         left=np.nan, right=np.nan)


    def values(self):
        return self._values


    def xCoord(self):
        return self._xCoord


    def yCoord(self):
        return self._yCoord


    def radialCoord(self):
        return np.hypot(self.xCoord(), self.yCoord())


    def xyStep(self):
        return self._step


    def xOriginInPx(self):
        return self._xOriginInPx


    def yOriginInPx(self):
        return self._yOriginInPx


    def interpolateInXY(self, x, y):
        return self._my_interp(x, y, spn=3)


    def _getRadialProfile(self):
        radProfile, radDistanceInPx= computeRadialProfile(
            self.values(), self.yOriginInPx(), self.xOriginInPx())
        return radProfile, radDistanceInPx * self.xyStep()


    def plotRadialProfile(self):
        import matplotlib.pyplot as plt
        y, x= self._getRadialProfile()
        plt.plot(x, y)
        plt.show()


    def extent(self):
        return [self.xCoord().min(), self.xCoord().max(),
                self.yCoord().min(), self.yCoord().max()]


    def _my_interp(self, x, y, spn=3):
        xs, ys = map(np.array, (x, y))
        z = np.zeros(xs.shape)
        for i, (x, y) in enumerate(zip(xs, ys)):
            # get the indices of the nearest x,y
            xi = np.argmin(np.abs(self.xCoord()[0, :] - x))
            yi = np.argmin(np.abs(self.yCoord()[:, 0] - y))
            xlo = max(xi - spn, 0)
            ylo = max(yi - spn, 0)
            xhi = min(xi + spn, self.xCoord()[0, :].size)
            yhi = min(yi + spn, self.yCoord()[:, 0].size)
            # make slices of X,Y,Z that are only a few items wide
            nX = self.xCoord()[ylo:yhi, xlo:xhi]
            nY = self.yCoord()[ylo:yhi, xlo:xhi]
            nZ = self.values()[ylo:yhi, xlo:xhi]
            if np.iscomplexobj(nZ):
                z[i]= self._interpComplex(nX, nY, nZ, x, y)
            else:
                z[i]= self._interpReal(nX, nY, nZ, x, y)
        return z


    def _interpReal(self, nX, nY, nZ, x, y):
        intp = interpolate.interp2d(nX, nY, nZ)
        return intp(x, y)[0]


    def _interpComplex(self, nX, nY, nZ, x, y):
        intpr = interpolate.interp2d(nX, nY, nZ.real)
        intpi = interpolate.interp2d(nX, nY, nZ.imag)
        return intpr(x, y)[0] + 1j * intpi(x, y)[0]



    def cutOutFromROI(self, xmin, xmax, ymin, ymax):
        xmin = np.argmin(np.abs(self.xCoord()[0, :] - xmin))
        xmax = np.argmin(np.abs(self.xCoord()[0, :] - xmax)) + 1
        ymin = np.argmin(np.abs(self.yCoord()[:, 0] - ymin))
        ymax = np.argmin(np.abs(self.yCoord()[:, 0] - ymax)) + 1
        xlo = max(xmin, 0)
        ylo = max(ymin, 0)
        xhi = min(xmax+ 1, self.xCoord()[0, :].size)
        yhi = min(ymax+ 1, self.yCoord()[:, 0].size)
        nX = self.xCoord()[ylo:yhi, xlo:xhi]
        nY = self.yCoord()[ylo:yhi, xlo:xhi]
        nZ = self.values()[ylo:yhi, xlo:xhi]
        return ScalarBidimensionalFunction(nZ, nX, nY)
