import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class ModalDecomposer(object):

    FIRST_ZERNIKE_MODE = ZernikeCoefficients.FIRST_ZERNIKE_MODE

    def __init__(self, n_zernike_modes):
        self._nZernikeModes = n_zernike_modes

    @cacheResult
    def synthZernikeRecFromSlopes(self, nModes, circular_mask, user_mask=None, dtype=float):
        if user_mask is None:
            user_mask = circular_mask
        assert isinstance(circular_mask, CircularMask), \
            'circular_mask argument must be of type CircularMask, instead is %s' % \
            circular_mask.__class__.__name__
        assert isinstance(user_mask, BaseMask), \
            'user_mask argument must be of type BaseMask, instead is %s' % \
            user_mask.__class__.__name__
        if not np.all(
                circular_mask.as_masked_array() * user_mask.as_masked_array()
                == user_mask.as_masked_array()):
            raise ValueError(
                'User mask must be fully contained in circular mask')

        zg = ZernikeGenerator(circular_mask)
        dx = zg.getDerivativeXDict(
            list(range(self.FIRST_ZERNIKE_MODE, self.FIRST_ZERNIKE_MODE + nModes)))
        dy = zg.getDerivativeYDict(
            list(range(self.FIRST_ZERNIKE_MODE, self.FIRST_ZERNIKE_MODE + nModes)))
        im = np.zeros(
            (nModes, 2*user_mask.as_masked_array().compressed().size), dtype=dtype)
        modesIdx = list(range(self.FIRST_ZERNIKE_MODE,
                        self.FIRST_ZERNIKE_MODE + nModes))

        for i, idx in enumerate(modesIdx):
            dx_masked = np.ma.masked_array(dx[idx].data, mask=user_mask.mask())
            dy_masked = np.ma.masked_array(dy[idx].data, mask=user_mask.mask())
            im[i, :] = np.hstack(
                (dx_masked.compressed(), dy_masked.compressed()))
        return pinv(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromSlopes(self, slopes, circular_mask,
                                             user_mask=None, nModes=None, dtype=float):
        if user_mask is None:
            user_mask = circular_mask
        if nModes is None:
            nModes = self._nZernikeModes
        assert isinstance(slopes, Slopes), \
            'slopes argument must be of type Slopes, instead is %s' % \
            slopes.__class__.__name__

        reconstructor = self.synthZernikeRecFromSlopes(
            nModes, circular_mask, user_mask, dtype)

        slopesInMaskVector = np.hstack(
            (np.ma.masked_array(slopes.mapX(), user_mask.mask()).compressed(),
             np.ma.masked_array(slopes.mapY(), user_mask.mask()).compressed())
        )
        
        return ZernikeCoefficients.fromNumpyArray(
            np.dot(slopesInMaskVector, reconstructor))

    @cacheResult
    def synthZernikeRecFromWavefront(self, nModes, circular_mask,
                                      user_mask, dtype=float):
        assert isinstance(circular_mask, CircularMask), \
            'Circular mask argument must be of type CircularMask, instead is %s' % \
            circular_mask.__class__.__name__
        assert isinstance(user_mask, BaseMask), \
            'User mask argument must be of type BaseMask, instead is %s' % \
            user_mask.__class__.__name__
        if not np.all(
            circular_mask.as_masked_array() * user_mask.as_masked_array()
            == user_mask.as_masked_array()):
            raise ValueError('User mask must be fully contained in circular mask')

        zg = ZernikeGenerator(circular_mask)
        wf = zg.getZernikeDict(
            list(range(self.FIRST_ZERNIKE_MODE, self.FIRST_ZERNIKE_MODE + nModes)))
        im = np.zeros((nModes, user_mask.as_masked_array().compressed().size), dtype=dtype)
        modesIdx = list(range(self.FIRST_ZERNIKE_MODE,
                        self.FIRST_ZERNIKE_MODE + nModes))

        for i, idx in enumerate(modesIdx):
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
            im[i,:] = wf_masked.compressed()
        return pinv(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(self, wavefront, circular_mask,
                                                user_mask=None, nModes=None, dtype=float):
        if user_mask is None:
            user_mask = circular_mask
        if nModes is None:
            nModes = self._nZernikeModes
        assert isinstance(wavefront, Wavefront), \
            'wavefront argument must be of type Wavefront, instead is %s' % \
            wavefront.__class__.__name__

        reconstructor = self.synthZernikeRecFromWavefront(nModes,
                                                           circular_mask,
                                                           user_mask, dtype)
        wavefrontInMaskVector = \
            np.ma.masked_array(wavefront.toNumpyArray(),
                               user_mask.mask()).compressed()
        wavefrontInMaskVectorNoPiston = wavefrontInMaskVector - \
            wavefrontInMaskVector.mean()
        return ZernikeCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, reconstructor))
