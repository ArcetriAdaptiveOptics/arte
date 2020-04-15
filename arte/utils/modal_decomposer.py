import numpy as np
from scipy.linalg.basic import pinv2
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class ModalDecomposer(object):

    def __init__(self, n_zernike_modes):
        self._nZernikeModes = n_zernike_modes

    @cacheResult
    def _synthZernikeRecFromSlopes(self, nModes, radiusInPixels):
        diameterInPixels = 2 * radiusInPixels
        zg = ZernikeGenerator(diameterInPixels)
        dx = zg.getDerivativeXDict(list(range(2, 2 + nModes)))
        dy = zg.getDerivativeYDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, 2 * dx[2].compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            im[i, :] = np.hstack((dx[idx].compressed(), dy[idx].compressed()))
            i += 1
        return pinv2(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromSlopes(self, slopes, mask, nModes=None):
        if nModes is None:
            nModes = self._nZernikeModes
        assert isinstance(slopes, Slopes), \
            'slopes argument must be of type Slopes, instead is %s' % \
            slopes.__class__.__name__
        assert isinstance(mask, CircularMask), \
            'Mask argument must be of type CircularMask, instead is %s' % \
            mask.__class__.__name__

        reconstructor = self._synthZernikeRecFromSlopes(nModes, mask.radius())

        slopesInMaskVector = np.hstack(
            (np.ma.masked_array(slopes.mapX(), mask.mask()).compressed(),
             np.ma.masked_array(slopes.mapY(), mask.mask()).compressed())
        )

        return ZernikeCoefficients.fromNumpyArray(
            np.dot(slopesInMaskVector, reconstructor))

    @cacheResult
    def _synthZernikeRecFromWavefront(self, nModes, radiusInPixels):
        diameterInPixels = 2 * radiusInPixels
        zg = ZernikeGenerator(diameterInPixels)
        wf = zg.getZernikeDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, wf[2].compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            im[i, :] = wf[idx].compressed()
            i += 1
        return pinv2(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(self,
                                                wavefront,
                                                mask,
                                                nModes=None):
        if nModes is None:
            nModes = self._nZernikeModes
        assert isinstance(wavefront, Wavefront), \
            'wavefront argument must be of type Wavefront, instead is %s' % \
            wavefront.__class__.__name__
        assert isinstance(mask, CircularMask), \
            'Mask argument must be of type CircularMask, instead is %s' % \
            mask.__class__.__name__

        reconstructor = self._synthZernikeRecFromWavefront(nModes,
                                                           mask.radius())
        wavefrontInMaskVector = \
            np.ma.masked_array(wavefront.toNumpyArray(),
                               mask.mask()).compressed()
        wavefrontInMaskVectorNoPiston = wavefrontInMaskVector - \
            wavefrontInMaskVector.mean()
        return ZernikeCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, reconstructor))
