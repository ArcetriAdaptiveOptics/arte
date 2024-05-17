import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes
from arte.utils.modal_decomposer import ModalDecomposer


# write the class ZernikeDecomposer here derived from ModalDecomposer and is a copy o that with only the name changed
class ZernikeModalDecomposer(ModalDecomposer):
    def __init__(self, nModes):
        super().__init__(nModes)


