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
        self._lastModesGenerator = None
        self._lastIM = None
        self._lastRank = None
        self._lastReconstructor = None
        self._lastMask = None

    @abc.abstractmethod
    def _generator(self, nModes, circular_mask, user_mask, **kwargs):
        '''Override to return a modal generator instance'''

    def _numpy2coefficients(self, coeff_array):
        '''Override to convert the resulting numpy array
        to a specific return type, if needed'''
        return ModalCoefficients(coeff_array)

    def getLastRank(self):
        return self._lastRank

    def _pistonModeIndex(self):
        '''
        Override to return the index of the piston (constant) mode of the
        modal basis, if it has one. When that mode is among the fitted
        modes the piston is measured; otherwise it is removed both from
        the wavefront and from the modes (default: no piston mode).
        '''
        return None

    def _removesPiston(self, modesIdx):
        return self._pistonModeIndex() not in modesIdx

    def _firstMode(self, start_mode):
        '''First mode index actually used for a given start_mode'''
        if start_mode is not None:
            return start_mode
        return self._lastModesGenerator.first_mode()

    def _startModeFromCoefficients(self, modal_coefficients):
        '''
        start_mode to be used to recompose a wavefront from
        modal_coefficients: the index of its first mode
        '''
        return modal_coefficients.FIRST_MODE

    def _wavefront_interaction_matrix(self, modes_generator, modesIdx, user_mask, dtype):
        wf = modes_generator.getModesDict(modesIdx)
        nslopes = user_mask.as_masked_array().compressed().size
        im = np.zeros((len(modesIdx), nslopes), dtype=dtype)
        remove_piston = self._removesPiston(modesIdx)
        for i, idx in enumerate(modesIdx):
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
            mode_compressed = wf_masked.compressed()
            if remove_piston:
                # Remove piston from each mode to match piston removal
                # in measurement
                mode_compressed = mode_compressed - mode_compressed.mean()
            im[i, :] = mode_compressed
        return im
    
    def _slopes_interaction_matrix(self, modes_generator, modesIdx, user_mask, dtype):

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

    def _syntheticInteractionMatrix(self, im_func, nModes,
                                    circular_mask, user_mask=None,
                                    dtype=float, start_mode=None,
                                    **kwargs):
        '''Generates a synthetic reconstructor given a certain IM function'''
        if user_mask is None:
            user_mask = circular_mask

        self._assert_types(circular_mask, user_mask)

        modes_generator = self._generator(
            nModes, circular_mask, user_mask, **kwargs)
        if start_mode is None:
            first_mode = modes_generator.first_mode()
        else:
            first_mode = start_mode

        self._lastModesGenerator = modes_generator

        modesIdx = list(range(first_mode, first_mode + nModes))
        im = im_func(modes_generator, modesIdx, user_mask, dtype)
        self._lastIM = im
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

        im = self._syntheticInteractionMatrix(im_func, nModes,
                                              circular_mask, user_mask=user_mask,
                                              dtype=dtype, start_mode=start_mode,
                                              **kwargs)
        return pinv(im, return_rank=return_rank, atol=atol, rtol=rtol, check_finite=check_finite)

    @functools.cache
    def cachedSyntheticInteractionMatrixFromWavefront(self, nModes,
                                                      circular_mask, user_mask=None,
                                                      dtype=float, start_mode=None,
                                                      **kwargs):
        '''Generates a synthetic interaction matrix instance using the
           modal basis returned by self.generator()'''
        return self._syntheticInteractionMatrix(self._wavefront_interaction_matrix, nModes,
                                                circular_mask, user_mask=user_mask,
                                                dtype=dtype, start_mode=start_mode, **kwargs)


    @functools.cache
    def cachedSyntheticReconstructorFromWavefront(self, nModes,
                                                  circular_mask, user_mask=None,
                                                  dtype=float, start_mode=None,
                                                  return_rank=False, **kwargs):
        '''Generates a synthetic reconstructor instance using the
           modal basis returned by self.generator()'''
        return self._syntheticReconstructor(self._wavefront_interaction_matrix, nModes,
                                            circular_mask, user_mask,
                                            dtype, start_mode, return_rank, **kwargs)

    @functools.cache
    def cachedSyntheticReconstructorFromSlopes(self, nModes,
                                                  circular_mask, user_mask=None,
                                                  dtype=float, start_mode=None,
                                                  return_rank=False, **kwargs):
        '''Generates a synthetic reconstructor instance using the
           modal basis returned by self.generator()'''
        return self._syntheticReconstructor(self._slopes_interaction_matrix, nModes,
                                            circular_mask, user_mask,
                                            dtype, start_mode, return_rank, **kwargs)

    def recomposeWavefrontFromModalCoefficients(
            self, modal_coefficients, circular_mask, dtype=float, **kwargs):
        '''
        The modes used are those labelled by modal_coefficients, i.e.
        modal_coefficients.modeIndexes().
        '''
        self._assert_types(
            circular_mask, modal_coefficients=modal_coefficients)
        nModes = modal_coefficients.numberOfModes()
        start_mode = self._startModeFromCoefficients(modal_coefficients)
        if kwargs.get('start_mode') is not None and \
                kwargs['start_mode'] != start_mode:
            raise ValueError(
                'start_mode=%s inconsistent with the first mode index (%s) '
                'of modal_coefficients' % (kwargs['start_mode'], start_mode))
        kwargs['start_mode'] = start_mode
        interaction_matrix = self.cachedSyntheticInteractionMatrixFromWavefront(
            nModes, circular_mask, circular_mask, dtype=dtype, return_rank=True, **kwargs
        )
        wf = np.dot(modal_coefficients.toNumpyArray(), interaction_matrix)
        wfm = np.ma.masked_array(
            np.zeros(circular_mask.shape()), mask=circular_mask.mask())
        wfm[~wfm.mask] = wf
        return Wavefront.fromNumpyArray(wfm)

    def measureModalCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, nModes=None, dtype=float, **kwargs
    ):
        self._assert_types(circular_mask, user_mask, wavefront=wavefront)
        if nModes is None:
            nModes = self.nModes

        reconstructor, rank = self.cachedSyntheticReconstructorFromWavefront(
            nModes, circular_mask, user_mask, dtype=dtype, return_rank=True, **kwargs
        )

        first_mode = self._firstMode(kwargs.get('start_mode'))
        modesIdx = list(range(first_mode, first_mode + nModes))
        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        if self._removesPiston(modesIdx):
            wavefrontInMaskVector = (
                wavefrontInMaskVector - wavefrontInMaskVector.mean()
            )
        result = self._numpy2coefficients(
            np.dot(wavefrontInMaskVector, reconstructor)
        )
        result.FIRST_MODE = first_mode
        # Remember last used values
        self._lastRank = rank
        self._lastMask = user_mask
        self._lastReconstructor = reconstructor
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
        result.FIRST_MODE = self._firstMode(None)
        # Remember last used values
        self._lastRank = rank
        self._lastMask = user_mask
        self._lastReconstructor = reconstructor
        return result

    def _assert_types(self, circular_mask, user_mask=None, wavefront=None,
                      slopes=None, modal_coefficients=None):
        '''
        Make sure that:
         1) circular_mask is of type CircularMask
         2) user_mask, if specified, is of type BaseMask
         3) user_mask, if specified, is fully contained into circular_mask
         4) wavefront, if specified, is of type Wavefront
         5) slopes, if specified, is of type Slopes
         6) modal_coefficients, if specified, is of type ModalCoefficient

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
        if modal_coefficients:
            assert isinstance(modal_coefficients, ModalCoefficients), (
                "modal_coefficients argument must be of type ModalCoefficients, instead is %s"
                % modal_coefficients.__class__.__name__
            )
