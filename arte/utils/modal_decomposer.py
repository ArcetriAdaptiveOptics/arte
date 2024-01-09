import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class ModalDecomposer(object):

    def __init__(self, n_zernike_modes):
        self._nZernikeModes = n_zernike_modes

    @cacheResult
    def _synthZernikeRecFromSlopes(self, nModes, circular_mask):
        zg = ZernikeGenerator(circular_mask)
        dx = zg.getDerivativeXDict(list(range(2, 2 + nModes)))
        dy = zg.getDerivativeYDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, 2 * dx[2].compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            im[i,:] = np.hstack((dx[idx].compressed(), dy[idx].compressed()))
            i += 1
        return pinv(im)

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

        reconstructor = self._synthZernikeRecFromSlopes(nModes, mask)

        slopesInMaskVector = np.hstack(
            (np.ma.masked_array(slopes.mapX(), mask.mask()).compressed(),
             np.ma.masked_array(slopes.mapY(), mask.mask()).compressed())
        )

        return ZernikeCoefficients.fromNumpyArray(
            np.dot(slopesInMaskVector, reconstructor))

    @cacheResult
    def _synthZernikeRecFromWavefront(self, nModes, circular_mask,
                                      user_mask):
        zg = ZernikeGenerator(circular_mask)
        wf = zg.getZernikeDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
            im[i,:] = wf_masked.compressed()
            i += 1
        return pinv(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(self,
                                                wavefront,
                                                circular_mask,
                                                user_mask,
                                                nModes=None):
        if nModes is None:
            nModes = self._nZernikeModes
        assert isinstance(wavefront, Wavefront), \
            'wavefront argument must be of type Wavefront, instead is %s' % \
            wavefront.__class__.__name__
        assert isinstance(circular_mask, CircularMask), \
            'Circular mask argument must be of type CircularMask, instead is %s' % \
            circular_mask.__class__.__name__
        assert isinstance(user_mask, BaseMask), \
            'User mask argument must be of type BaseMask, instead is %s' % \
            user_mask.__class__.__name__
        if not np.all(
            circular_mask.as_masked_array() * user_mask.as_masked_array()
            == user_mask.as_masked_array()):
            raise Exception('User mask must be fully contained in circular mask')

        reconstructor = self._synthZernikeRecFromWavefront(nModes,
                                                           circular_mask,
                                                           user_mask)
        wavefrontInMaskVector = \
            np.ma.masked_array(wavefront.toNumpyArray(),
                               user_mask.mask()).compressed()
        wavefrontInMaskVectorNoPiston = wavefrontInMaskVector - \
            wavefrontInMaskVector.mean()
        return ZernikeCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, reconstructor))
