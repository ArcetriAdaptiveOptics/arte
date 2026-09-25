import warnings
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.utils.base_modal_decomposer import BaseModalDecomposer
from arte.utils.decorator import returns


class ZernikeModalDecomposer(BaseModalDecomposer):
    """
    This class decomposes a wavefront or slope array into a set of modal zernike coefficients.

    Parameters
    ----------
    n_modes: int
        Number of modes to decompose the wavefront into.

    Notes
    -----
    By default the modes are Z2, Z3, ... (Noll). Pass mode_indexes to the
    measure methods to choose any set of Zernike modes; if it contains
    piston (Z1), piston is fitted instead of being removed.
    """

    DEFAULT_FIRST_MODE = 2
    PISTON_MODE_INDEX = 1

    def __init__(self, n_modes=None, n_zernike_modes=None):
        if n_modes is None and n_zernike_modes is None:
            raise ValueError("either n_modes or n_zernike_modes must be specified")
        if n_zernike_modes is not None:
            n_modes = n_zernike_modes
        super().__init__(n_modes)

    def _generator(self, nModes, circular_mask, user_mask, **kwargs):
        return ZernikeGenerator(circular_mask)

    def _numpy2coefficients(self, coeff_array, mode_indexes=None):
        return ZernikeCoefficients(coeff_array, mode_indexes=mode_indexes)

    def _recomposeModeIndexes(self, modal_coefficients):
        if isinstance(modal_coefficients, ZernikeCoefficients):
            return modal_coefficients.zernikeIndexes()
        # Backward compatibility: generic ModalCoefficients have no Zernike
        # labels and are interpreted as Z2, Z3, ...
        warnings.warn(
            'recomposing generic ModalCoefficients as Z%d, Z%d, ...: use '
            'ZernikeCoefficients to choose the Zernike modes' % (
                self.DEFAULT_FIRST_MODE, self.DEFAULT_FIRST_MODE + 1),
            DeprecationWarning, stacklevel=3)
        return self._resolveModeIndexes(modal_coefficients.numberOfModes())


    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(self, wavefront, circular_mask,
                                                user_mask=None, nModes=None, dtype=float,
                                                mode_indexes=None):
        return self.measureModalCoefficientsFromWavefront(wavefront, circular_mask,
                                                       user_mask, nModes, dtype=dtype,
                                                       mode_indexes=mode_indexes)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromSlopes(self, slopes, circular_mask,
                                             user_mask=None, nModes=None, dtype=float,
                                             mode_indexes=None):
        return self.measureModalCoefficientsFromSlopes(slopes, circular_mask,
                                                       user_mask, nModes, dtype=dtype,
                                                       mode_indexes=mode_indexes)

    def synthZernikeRecFromSlopes(self, nModes, circular_mask, user_mask=None, dtype=float):
        return self.cachedSyntheticReconstructorFromSlopes(nModes, circular_mask,
                                                           user_mask, dtype=dtype)

    def synthZernikeRecFromWavefront(self, nModes, circular_mask, user_mask=None, dtype=float):
        return self.cachedSyntheticReconstructorFromWavefront(nModes, circular_mask,
                                                           user_mask, dtype=dtype)
