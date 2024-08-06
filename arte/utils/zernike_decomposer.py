from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.utils.base_modal_decomposer import BaseModalDecomposer


class ZernikeModalDecomposer(BaseModalDecomposer):
    """
    This class decomposes a wavefront or slope array into a set of modal zernike coefficients.

    Parameters
    ----------
    n_modes: int
        Number of modes to decompose the wavefront into.
    """

    def __init__(self, n_modes=None, n_zernike_modes=None):
        if n_modes is None and n_zernike_modes is None:
            raise ValueError("either n_modes or n_zernike_modes must be specified")
        if n_zernike_modes is not None:
            n_modes = n_zernike_modes
        super().__init__(n_modes)

    def _generator(self, nModes, circular_mask, user_mask, **kwargs):
        return ZernikeGenerator(circular_mask)

    def _numpy2coefficients(self, coeff_array):
        return ZernikeCoefficients(coeff_array)



