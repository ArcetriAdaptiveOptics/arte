import numpy as np
from arte.utils.rbf_generator import RBFGenerator
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront
from arte.utils.base_modal_decomposer import BaseModalDecomposer


class RadialBasisModalDecomposer(BaseModalDecomposer):

    # write class and method documentation here
    """
    This class decomposes a wavefront into a set of modal coefficients. The modal coefficients can be
    Zernike, KL or RBF coefficients. The class provides methods to measure the modal coefficients from
    slopes or wavefronts. The methods are:
    - measureZernikeCoefficientsFromSlopes
    - measureZernikeCoefficientsFromWavefront
    - measureKLCoefficientsFromWavefront
    - measureModalCoefficientsFromWavefront
    Parameters
    ----------
    n_modes: int
        Number of modes to decompose the wavefront into.
    """

    def __init__(self, coordinates_list):
        self.coordinates_list = coordinates_list
        super().__init__(n_modes = len(coordinates_list))

    def _generator(self, nModes, circular_mask, user_mask, rbfFunction=None, **kwargs):
        rbf = RBFGenerator(circular_mask, self.coordinates_list, rbfFunction=rbfFunction)
        rbf.generate()
        return rbf

