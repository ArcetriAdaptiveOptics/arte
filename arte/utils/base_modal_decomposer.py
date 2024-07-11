import abc
import functools
import numpy as np
from scipy.linalg import pinv
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class BaseModalDecomposer(abc.ABC):
    """
    Generic modal decomposer class.
    """

    def __init__(self, n_modes=None):
        self.nModes = n_modes
        self.lastModesGenerator = None
        self.lastIM = None
        self.lastRank = None
        self.lastReconstructor = None
        self.lastMask = None

    @abc.abstractmethod
    def generator(self, nModes, circular_mask, user_mask, **kwargs):
        '''Override to return a modal generator instance'''

    def _numpy2coefficients(self, coeff_array):
        '''Override to convert the resulting numpy array
        to a specific return type, if needed'''
        return ModalCoefficients(coeff_array)

    def getLastRank(self):
        return self.lastRank

    def wfIm(self, modes_generator, modesIdx, user_mask, dtype):
        wf = modes_generator.getModesDict(modesIdx)
        nslopes = user_mask.as_masked_array().compressed().size
        im = np.zeros((len(modesIdx), nslopes), dtype=dtype)
        for i, idx in enumerate(modesIdx):
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
            im[i, :] = wf_masked.compressed()
        return im
    
    def slopesIm(self, modes_generator, modesIdx, user_mask, dtype):

        if not hasattr(modes_generator, 'getDerivativeXDict') or \
           not hasattr(modes_generator, 'getDerivativeYDict'):
            raise NotImplementedError(f'Modes generator of type {modes_generator.__class__.__name__}'
                                      ' does not define methods to get derivatives')

        dx = modes_generator.getDerivativeXDict(modesIdx)
        dy = modes_generator.getDerivativeYDict(modesIdx)
        nslopes = user_mask.as_masked_array().compressed().size
        im = np.zeros((len(modesIdx), 2 * nslopes), dtype=dtype)

        for i, idx in enumerate(modesIdx):
            dx_masked = np.ma.masked_array(dx[idx].data, mask=user_mask.mask())
            dy_masked = np.ma.masked_array(dy[idx].data, mask=user_mask.mask())
            im[i, :] = np.hstack(
                (dx_masked.compressed(), dy_masked.compressed()))
        return im

    def _syntheticReconstructor(self, im_func, nModes,
                                circular_mask, user_mask=None,
                                dtype=float, start_mode=None,
                                return_rank=False, **kwargs):
        '''Generates a synthetic reconstructor given a certain IM function'''
        if user_mask is None:
            user_mask = circular_mask

        atol = kwargs.pop('atol', None)
        rtol = kwargs.pop('rtol', None)
        check_finite = kwargs.pop('check_finite', True)

        self._assert_types(circular_mask, user_mask)

        modes_generator = self.generator(nModes, circular_mask, user_mask, **kwargs)
        if start_mode is None:
            first_mode = modes_generator.first_mode()
        else:
            first_mode = start_mode

        self.lastModesGenerator = modes_generator

        modesIdx = list(range(first_mode, first_mode + nModes))
        im = im_func(modes_generator, modesIdx, user_mask, dtype)
        self.lastIM = im
        return pinv(im, return_rank=return_rank, atol=atol, rtol=rtol, check_finite=check_finite)

    @functools.cache
    def cachedSyntheticReconstructorFromWavefront(self, nModes,
                                                  circular_mask, user_mask=None,
                                                  dtype=float, start_mode=None,
                                                  return_rank=False, **kwargs):
        '''Generates a synthetic reconstructor instance using the
           modal basis returned by self.generator()'''
        return self._syntheticReconstructor(self.wfIm, nModes, 
                                            circular_mask, user_mask,
                                            dtype, start_mode, return_rank, **kwargs)

    @functools.cache
    def cachedSyntheticReconstructorFromSlopes(self, nModes,
                                                  circular_mask, user_mask=None,
                                                  dtype=float, start_mode=None,
                                                  return_rank=False, **kwargs):
        '''Generates a synthetic reconstructor instance using the
           modal basis returned by self.generator()'''
        return self._syntheticReconstructor(self.slopesIm, nModes,
                                            circular_mask, user_mask,
                                            dtype, start_mode, return_rank, **kwargs)

    def measureModalCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, nModes=None, dtype=float, **kwargs
    ):
        self._assert_types(circular_mask, user_mask, wavefront=wavefront)
        if nModes is None:
            nModes = self.nModes

        reconstructor, rank = self.cachedSyntheticReconstructorFromWavefront(
            nModes, circular_mask, user_mask, dtype=dtype, return_rank=True, **kwargs
        )

        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        wavefrontInMaskVectorNoPiston = (
            wavefrontInMaskVector - wavefrontInMaskVector.mean()
        )
        result = self._numpy2coefficients(
            np.dot(wavefrontInMaskVectorNoPiston, reconstructor)
        )
        # Remember last used values
        self.lastRank = rank
        self.lastMask = user_mask
        self.lastReconstructor = reconstructor
        return result

    def measureModalCoefficientsFromSlopes(self, slopes, circular_mask,
                                           user_mask=None, nModes=None, dtype=float):

        self._assert_types(circular_mask, user_mask, slopes=slopes)
        if nModes is None:
            nModes = self.nModes
        if user_mask is None:
            user_mask = circular_mask

        reconstructor, rank = self.cachedSyntheticReconstructorFromSlopes(
            nModes, circular_mask, user_mask, dtype=dtype, return_rank=True)

        slopesInMaskVector = np.hstack(
            (np.ma.masked_array(slopes.mapX(), user_mask.mask()).compressed(),
             np.ma.masked_array(slopes.mapY(), user_mask.mask()).compressed())
        )

        result = self._numpy2coefficients(np.dot(slopesInMaskVector, reconstructor))
        # Remember last used values
        self.lastRank = rank
        self.lastMask = user_mask
        self.lastReconstructor = reconstructor
        return result

    def _assert_types(self, circular_mask, user_mask=None, wavefront=None, slopes=None):
        '''
        Make sure that:
         1) circular_mask is of type CircularMask
         2) user_mask, if specified, is of type BaseMask
         3) user_mask, if specified, is fully contained into circular_mask
         4) wavefront, if specified, is of type Wavefront
         5) slopes, if specified, is of type Slopes

        Raise an AssertionError if not.
        '''
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

        if wavefront:
            assert isinstance(wavefront, Wavefront), (
                "wavefront argument must be of type Wavefront, instead is %s"
                % wavefront.__class__.__name__
            )
        if slopes:
            assert isinstance(slopes, Slopes), (
                "slopes argument must be of type Slopes, instead is %s"
                % slopes.__class__.__name__
            )
