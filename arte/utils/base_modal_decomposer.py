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

    The modes of a decomposition are chosen, in order of precedence, by:

    - mode_indexes: any sequence of mode indexes, e.g. [2, 3, 11];
    - nModes and start_mode: modes start_mode, ..., start_mode + nModes - 1;
    - nModes only: the same, starting from DEFAULT_FIRST_MODE.

    The measured coefficients are labelled by those mode indexes.
    If PISTON_MODE_INDEX is among them, piston is fitted as any other
    mode; otherwise it is removed from both the wavefront and the modes.
    """

    DEFAULT_FIRST_MODE = 0
    '''Index of the first mode when neither start_mode nor mode_indexes
    are given'''

    PISTON_MODE_INDEX = None
    '''Index of the piston (constant) mode of the modal basis, if any'''

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

    def _numpy2coefficients(self, coeff_array, mode_indexes=None):
        '''Override to convert the resulting numpy array
        to a specific return type, if needed'''
        return ModalCoefficients(coeff_array, mode_indexes=mode_indexes)

    def getLastRank(self):
        return self._lastRank

    def _resolveModeIndexes(self, nModes, start_mode=None, mode_indexes=None):
        '''
        Return the mode indexes of a decomposition, as a tuple of int.

        If mode_indexes is given, start_mode must be None and nModes,
        if not None, must be equal to len(mode_indexes).
        '''
        if mode_indexes is not None:
            if start_mode is not None:
                raise ValueError(
                    'start_mode and mode_indexes cannot be both specified')
            idx = ModalCoefficients._checkModeIndexes(
                mode_indexes, len(np.atleast_1d(mode_indexes)))
            if nModes is not None and nModes != len(idx):
                raise ValueError(
                    'nModes=%d inconsistent with %d mode_indexes' % (
                        nModes, len(idx)))
            return tuple(int(i) for i in idx)
        if nModes is None:
            raise ValueError('either nModes or mode_indexes must be specified')
        if start_mode is None:
            start_mode = self.DEFAULT_FIRST_MODE
        return tuple(range(start_mode, start_mode + nModes))

    def _removesPiston(self, modesIdx):
        return self.PISTON_MODE_INDEX not in modesIdx

    def _recomposeModeIndexes(self, modal_coefficients):
        '''Mode indexes of the modes used to recompose modal_coefficients'''
        return modal_coefficients.modeIndexes()

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
                                    mode_indexes=None, **kwargs):
        '''Generates a synthetic interaction matrix given a certain IM function'''
        if user_mask is None:
            user_mask = circular_mask

        self._assert_types(circular_mask, user_mask)

        modesIdx = list(self._resolveModeIndexes(nModes, start_mode, mode_indexes))
        modes_generator = self._generator(
            len(modesIdx), circular_mask, user_mask, **kwargs)
        self._lastModesGenerator = modes_generator

        im = im_func(modes_generator, modesIdx, user_mask, dtype)
        self._lastIM = im
        return im

    def _syntheticReconstructor(self, im_func, nModes,
                                circular_mask, user_mask=None,
                                dtype=float, start_mode=None,
                                return_rank=False, mode_indexes=None, **kwargs):
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
                                              mode_indexes=mode_indexes, **kwargs)
        return pinv(im, return_rank=return_rank, atol=atol, rtol=rtol, check_finite=check_finite)

    @functools.cache
    def cachedSyntheticInteractionMatrixFromWavefront(self, nModes,
                                                      circular_mask, user_mask=None,
                                                      dtype=float, start_mode=None,
                                                      mode_indexes=None, **kwargs):
        '''Generates a synthetic interaction matrix instance using the
           modal basis returned by self.generator().
           mode_indexes, if given, must be hashable (e.g. a tuple).'''
        return self._syntheticInteractionMatrix(self._wavefront_interaction_matrix, nModes,
                                                circular_mask, user_mask=user_mask,
                                                dtype=dtype, start_mode=start_mode,
                                                mode_indexes=mode_indexes, **kwargs)


    @functools.cache
    def cachedSyntheticReconstructorFromWavefront(self, nModes,
                                                  circular_mask, user_mask=None,
                                                  dtype=float, start_mode=None,
                                                  return_rank=False, mode_indexes=None,
                                                  **kwargs):
        '''Generates a synthetic reconstructor instance using the
           modal basis returned by self.generator().
           mode_indexes, if given, must be hashable (e.g. a tuple).'''
        return self._syntheticReconstructor(self._wavefront_interaction_matrix, nModes,
                                            circular_mask, user_mask,
                                            dtype, start_mode, return_rank,
                                            mode_indexes=mode_indexes, **kwargs)

    @functools.cache
    def cachedSyntheticReconstructorFromSlopes(self, nModes,
                                                  circular_mask, user_mask=None,
                                                  dtype=float, start_mode=None,
                                                  return_rank=False, mode_indexes=None,
                                                  **kwargs):
        '''Generates a synthetic reconstructor instance using the
           modal basis returned by self.generator().
           mode_indexes, if given, must be hashable (e.g. a tuple).'''
        return self._syntheticReconstructor(self._slopes_interaction_matrix, nModes,
                                            circular_mask, user_mask,
                                            dtype, start_mode, return_rank,
                                            mode_indexes=mode_indexes, **kwargs)

    def recomposeWavefrontFromModalCoefficients(
            self, modal_coefficients, circular_mask, dtype=float, **kwargs):
        '''
        Wavefront sum of the modes labelled by
        modal_coefficients.modeIndexes(), weighted by the coefficients.
        '''
        self._assert_types(
            circular_mask, modal_coefficients=modal_coefficients)
        modesIdx = tuple(int(i) for i in
                         self._recomposeModeIndexes(modal_coefficients))
        start_mode = kwargs.pop('start_mode', None)
        mode_indexes = kwargs.pop('mode_indexes', None)
        if start_mode is not None or mode_indexes is not None:
            if self._resolveModeIndexes(
                    len(modesIdx), start_mode, mode_indexes) != modesIdx:
                raise ValueError(
                    'start_mode/mode_indexes inconsistent with the mode '
                    'indexes of modal_coefficients %s' % (modesIdx,))
        interaction_matrix = self.cachedSyntheticInteractionMatrixFromWavefront(
            len(modesIdx), circular_mask, circular_mask, dtype=dtype, return_rank=True,
            mode_indexes=modesIdx, **kwargs
        )
        wf = np.dot(modal_coefficients.toNumpyArray(), interaction_matrix)
        wfm = np.ma.masked_array(
            np.zeros(circular_mask.shape()), mask=circular_mask.mask())
        wfm[~wfm.mask] = wf
        return Wavefront.fromNumpyArray(wfm)

    def measureModalCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, nModes=None, dtype=float,
        mode_indexes=None, **kwargs
    ):
        '''
        The modes are selected by mode_indexes, or by nModes (default:
        the n_modes given to the constructor) and the optional
        start_mode keyword. See the class documentation.
        '''
        self._assert_types(circular_mask, user_mask, wavefront=wavefront)
        if user_mask is None:
            user_mask = circular_mask
        if nModes is None and mode_indexes is None:
            nModes = self.nModes
        modesIdx = self._resolveModeIndexes(
            nModes, kwargs.pop('start_mode', None), mode_indexes)

        reconstructor, rank = self.cachedSyntheticReconstructorFromWavefront(
            len(modesIdx), circular_mask, user_mask, dtype=dtype, return_rank=True,
            mode_indexes=modesIdx, **kwargs
        )

        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        if self._removesPiston(modesIdx):
            wavefrontInMaskVector = (
                wavefrontInMaskVector - wavefrontInMaskVector.mean()
            )
        result = self._numpy2coefficients(
            np.dot(wavefrontInMaskVector, reconstructor), mode_indexes=modesIdx
        )
        # Remember last used values
        self._lastRank = rank
        self._lastMask = user_mask
        self._lastReconstructor = reconstructor
        return result

    def measureModalCoefficientsFromSlopes(self, slopes, circular_mask,
                                           user_mask=None, nModes=None, dtype=float,
                                           mode_indexes=None):
        '''
        The modes are selected by mode_indexes or by nModes (default:
        the n_modes given to the constructor), starting from
        DEFAULT_FIRST_MODE.
        '''
        self._assert_types(circular_mask, user_mask, slopes=slopes)
        if nModes is None and mode_indexes is None:
            nModes = self.nModes
        if user_mask is None:
            user_mask = circular_mask
        modesIdx = self._resolveModeIndexes(nModes, None, mode_indexes)

        reconstructor, rank = self.cachedSyntheticReconstructorFromSlopes(
            len(modesIdx), circular_mask, user_mask, dtype=dtype, return_rank=True,
            mode_indexes=modesIdx)

        slopesInMaskVector = np.hstack(
            (np.ma.masked_array(slopes.mapX(), user_mask.mask()).compressed(),
             np.ma.masked_array(slopes.mapY(), user_mask.mask()).compressed())
        )

        result = self._numpy2coefficients(np.dot(slopesInMaskVector, reconstructor),
                                          mode_indexes=modesIdx)
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
