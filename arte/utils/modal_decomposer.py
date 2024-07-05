from warnings import warn
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.utils.decorator import returns
from arte.utils.zernike_decomposer import ZernikeModalDecomposer


class ModalDecomposer(ZernikeModalDecomposer):
    '''
    Backward compatibility with old ModalDecomposer class
    '''
    def __init__(self, n_modes=None, n_zernike_modes=None):
        if n_zernike_modes is not None:
            n_modes = n_zernike_modes
        if n_modes is None and n_zernike_modes is None:
            raise ValueError("either n_modes or n_zernike_modes must be specified")
        super().__init__(n_modes)
        warn(
            f"{self.__class__.__name__} will be deprecated or change use. Use ZernikeModalDecomposer instead",
            DeprecationWarning,
            stacklevel=2,
        )

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(self, wavefront, circular_mask,
                                                user_mask=None, nModes=None, dtype=float):
        return self.measureModalCoefficientsFromWavefront(wavefront, circular_mask,
                                                       user_mask, nModes, dtype=dtype)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromSlopes(self, slopes, circular_mask,
                                             user_mask=None, nModes=None, dtype=float):
        return self.measureModalCoefficientsFromSlopes(slopes, circular_mask,
                                                       user_mask, nModes, dtype=dtype)

    def synthZernikeRecFromSlopes(self, nModes, circular_mask, user_mask=None, dtype=float):
        return self.cachedSyntheticReconstructorFromSlopes(nModes, circular_mask,
                                                           user_mask, dtype=dtype)

    def synthZernikeRecFromWavefront(self, nModes, circular_mask, user_mask=None, dtype=float):
        return self.cachedSyntheticReconstructorFromWavefront(nModes, circular_mask,
                                                           user_mask, dtype=dtype)
