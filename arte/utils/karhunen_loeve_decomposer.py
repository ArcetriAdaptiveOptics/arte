import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.utils.karhunen_loeve_generator import KarhunenLoeveGenerator as  KLGenerator
from arte.utils.base_modal_decomposer import BaseModalDecomposer


class KarhunenLoeveModalDecomposer(BaseModalDecomposer):
    """
    This class decomposes a wavefront into a set of Karhunen Loeve modal coefficients.

    Parameters
    ----------
    n_modes: int
        Number of modes to decompose the wavefront into.
    """

    def __init__(self, n_modes=None):
        super().__init__(n_modes)

    def _generator(self, nModes, circular_mask, user_mask, **kwargs):
        zz = ZernikeGenerator(circular_mask)
        zbase = np.rollaxis(
            np.ma.masked_array([zz.getZernike(n) for n in range(2, self.nModes + 2)]),
            0,
            3,
        )
        kl = KLGenerator(circular_mask, getFullKolmogorovCovarianceMatrix(self.nModes))
        kl.generateFromBase(zbase)
        return kl

