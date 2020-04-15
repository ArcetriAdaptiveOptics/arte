import numpy as np
from arte.types.scalar_bidimensional_function import \
    ScalarBidimensionalFunction
from arte.utils.coordinates import xCoordinatesMap


__version__= "$Id: $"


class BidimensionalFourierTransform(object):


    @staticmethod
    def distancesXMap(sizeInPoints, pixelSize):
        return xCoordinatesMap(sizeInPoints, pixelSize)


    @staticmethod
    def distancesYMap(sizeInPoints, pixelSize):
        return xCoordinatesMap(sizeInPoints, pixelSize).T


    @staticmethod
    def distancesNormMap(sizeInPoints, pixelSize):
        distX= BidimensionalFourierTransform.distancesXMap(
            sizeInPoints, pixelSize)
        distY= BidimensionalFourierTransform.distancesYMap(
            sizeInPoints, pixelSize)
        return np.linalg.norm(np.dstack((distX, distY)), axis=2)


    @staticmethod
    def frequenciesNormMap(sizeInPoints, pixelSize):
        n= sizeInPoints
        a=np.tile(np.fft.fftfreq(n, d=pixelSize), (n, 1))
        return np.fft.fftshift(np.linalg.norm(np.dstack((a, a.T)), axis=2))


    @staticmethod
    def frequenciesXMap(sizeInPoints, pixelSize):
        n= sizeInPoints
        a=np.tile(np.fft.fftfreq(n, d=pixelSize), (n, 1))
        return np.fft.fftshift(a)


    @staticmethod
    def frequenciesYMap(sizeInPoints, pixelSize):
        return BidimensionalFourierTransform.frequenciesXMap(
            sizeInPoints, pixelSize).T


    @staticmethod
    def _isEven(num):
        if num % 2 == 0:
            return True
        else:
            return False

    @staticmethod
    def mostPositiveFrequency(sizeInPoints, pixelSize):
        sizeInLengthUnits= sizeInPoints * pixelSize
        if BidimensionalFourierTransform._isEven(sizeInPoints):
            return (0.5* sizeInPoints -1) / sizeInLengthUnits
        else:
            return 0.5 * (sizeInPoints -1) / sizeInLengthUnits


    @staticmethod
    def mostNegativeFrequency(sizeInPoints, pixelSize):
        sizeInLengthUnits= sizeInPoints * pixelSize
        if BidimensionalFourierTransform._isEven(sizeInPoints):
            return -0.5* sizeInPoints / sizeInLengthUnits
        else:
            return -0.5 * (sizeInPoints -1) / sizeInLengthUnits


    @staticmethod
    def smallestFrequency(sizeInPoints, pixelSize):
        sizeInLengthUnits= sizeInPoints * pixelSize
        return 1./ sizeInLengthUnits


    @staticmethod
    def directTransform(data):
        res= np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(data, axes=(-1, -2)),
                norm="ortho"),
            axes=(-1, -2))
        return res


    @staticmethod
    def inverseTransform(data):
        res= np.fft.ifftshift(
            np.fft.ifft2(
                np.fft.ifftshift(data),
                norm="ortho"))
        return res

    @staticmethod
    def direct(xyFunct):
        sizeInPx= xyFunct.values().shape[0]
        pxSize= xyFunct.xyStep()
        return ScalarBidimensionalFunction(
            BidimensionalFourierTransform.directTransform(xyFunct.values()),
            BidimensionalFourierTransform.frequenciesXMap(sizeInPx, pxSize),
            BidimensionalFourierTransform.frequenciesYMap(sizeInPx, pxSize),
        )


    @staticmethod
    def inverse(xyFunct):
        sizeInPx= xyFunct.values().shape[0]
        pxSize= xyFunct.xyStep()
        return ScalarBidimensionalFunction(
            BidimensionalFourierTransform.inverseTransform(xyFunct.values()),
            BidimensionalFourierTransform.frequenciesXMap(sizeInPx, pxSize),
            BidimensionalFourierTransform.frequenciesYMap(sizeInPx, pxSize),
        )
        

#     @staticmethod
#     def directRealTransform(data, spacing):
#         res= np.fft.fftshift(
#             np.fft.rfft2(
#                 np.fft.fftshift(data, axes=(-1, -2))),
#             axes=(-1, -2)) * spacing**2
#         return res
# 
# 
#     @staticmethod
#     def inverseRealTransform(data, frequencySpacing):
#         sz= data.shape[0]
#         res= np.fft.ifftshift(
#             np.fft.irfft2(
#                 np.fft.ifftshift(data))) * (sz * frequencySpacing)**2
#         return res

