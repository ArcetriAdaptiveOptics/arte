import numpy as np
from arte.types.scalar_bidimensional_function import ScalarBidimensionalFunction
from arte.types.domainxy import DomainXY


class BidimensionalFourierTransform(object):

    @staticmethod
    def distances_x_map(sizeInPoints, pixelSize):
        domain = DomainXY.from_shape((sizeInPoints, sizeInPoints), pixelSize)
        return domain.xcoord

    @staticmethod
    def distances_norm_map(sizeInPoints, pixelSize):
        domain = DomainXY.from_shape((sizeInPoints, sizeInPoints), pixelSize)
        return np.linalg.norm(np.dstack((domain.xmap, domain.ymap)), axis=2)

    @staticmethod
    def frequencies_norm_map(sizeInPoints, pixelSize):
        n = sizeInPoints
        a = np.tile(np.fft.fftfreq(n, d=pixelSize), (n, 1))
        return np.fft.fftshift(np.linalg.norm(np.dstack((a, a.T)), axis=2))

    @staticmethod
    def frequencies_x_map(sizeInPoints, pixelSize):
        n = sizeInPoints
        a = np.tile(np.fft.fftfreq(n, d=pixelSize), (n, 1))
        return np.fft.fftshift(a)

    @staticmethod
    def frequencies_y_map(sizeInPoints, pixelSize):
        return BidimensionalFourierTransform.frequencies_x_map(
            sizeInPoints, pixelSize).T

    @staticmethod
    def _isEven(num):
        if num % 2 == 0:
            return True
        else:
            return False

    @staticmethod
    def most_positive_frequency(sizeInPoints, pixelSize):
        sizeInLengthUnits = sizeInPoints * pixelSize
        if BidimensionalFourierTransform._isEven(sizeInPoints):
            return (0.5 * sizeInPoints - 1) / sizeInLengthUnits
        else:
            return 0.5 * (sizeInPoints - 1) / sizeInLengthUnits

    @staticmethod
    def most_negative_frequency(sizeInPoints, pixelSize):
        sizeInLengthUnits = sizeInPoints * pixelSize
        if BidimensionalFourierTransform._isEven(sizeInPoints):
            return -0.5 * sizeInPoints / sizeInLengthUnits
        else:
            return -0.5 * (sizeInPoints - 1) / sizeInLengthUnits

    @staticmethod
    def smallest_frequency(sizeInPoints, pixelSize):
        sizeInLengthUnits = sizeInPoints * pixelSize
        return 1. / sizeInLengthUnits

    @staticmethod
    def direct_transform(data):
        res = np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(data, axes=(-1, -2)),
                norm="ortho"),
            axes=(-1, -2))
        return res

    @staticmethod
    def inverse_transform(data):
        res = np.fft.ifftshift(
            np.fft.ifft2(
                np.fft.ifftshift(data),
                norm="ortho"))
        return res

    @staticmethod
    def direct(xyFunct):
        sizeY, sizeX = xyFunct.values.shape
        pxSizeX, pxSizeY = xyFunct.domain.step
        return ScalarBidimensionalFunction(
            BidimensionalFourierTransform.direct_transform(
                xyFunct.values),
            xmap=BidimensionalFourierTransform.frequencies_x_map(
                sizeX, pxSizeX),
            ymap=BidimensionalFourierTransform.frequencies_y_map(
                sizeY, pxSizeY)
        )

    @staticmethod
    def reverse(xyFunct):
        sizeY, sizeX = xyFunct.values.shape
        pxSizeX, pxSizeY = xyFunct.domain.step
        return ScalarBidimensionalFunction(
            BidimensionalFourierTransform.inverse_transform(
                xyFunct.values),
            xmap=BidimensionalFourierTransform.frequencies_x_map(
                sizeX, pxSizeX),
            ymap=BidimensionalFourierTransform.frequencies_y_map(
                sizeY, pxSizeY)
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

