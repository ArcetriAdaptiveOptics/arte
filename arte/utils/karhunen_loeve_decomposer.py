import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.utils.karhunen_loeve_generator import KarhunenLoeveGenerator as  KLGenerator
from arte.utils.rbf_generator import RBFGenerator
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class KarhunenLoeveModalDecomposer(object):

    # write class and method documentation here
    """
    This class decomposes a wavefront into a set of Karhunen Loeve modal coefficients.
    The class provides methods to measure the modal coefficients from
    wavefronts. The methods are:
    - measureKLCoefficientsFromWavefront

    Parameters
    ----------
    n_modes: int
        Number of modes to decompose the wavefront into.
    covariance_matrix: ndarray
    """

    def __init__(self, n_modes=None, n_zernike_modes=None):
        self.nModes = n_modes
        if n_zernike_modes is not None:
            self.nModes = n_zernike_modes
        if n_modes is None and n_zernike_modes is None:
            raise ValueError("n_modes must be specified")
        self.reconstructor = None
        self.baseName = None
        self.baseModes = None
        self.baseDecomposition = None
        self.rank = None
        self.mask = None
        self.modesGenerator = None

    def getModalBase(self):
        return self.base

    def getModalReconstructor(self):
        return self.reconstructor

    def getModalBaseDecomposition(self):
        return self.baseDecomposition

    def getRank(self):
        return self.rank

    def mask(self):
        return self.mask


    # Write measureKLCoefficientsFromWavefront method here
    @cacheResult
    def _synthModeRecFromWavefront(
        self
    ):
        self.baseName = self.base
        zz = ZernikeGenerator(self.circular_mask)
        zbase = np.rollaxis(
            np.ma.masked_array([zz.getZernike(n) for n in range(2, self.nModes + 2)]),
            0,
            3,
        )
        kl = KLGenerator(self.circular_mask, getFullKolmogorovCovarianceMatrix(self.nModes))
        kl.generateFromBase(zbase)
        self.modesGenerator = kl
        wf = kl.getKLDict(list(range(self.start_mode, self.start_mode + self.nModes)))

        im = np.zeros((self.nModes, self.user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(self.start_mode, self.start_mode + self.nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=self.user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True)

    def _uncachedSynthModeRecFromWavefront(
        self, **kwargs
    ):
        self.baseName = self.base
        zz = ZernikeGenerator(self.circular_mask)
        zbase = np.rollaxis(
            np.ma.masked_array([zz.getZernike(n) for n in range(2, self.nModes + 2)]),
            0,
            3,
        )
        kl = KLGenerator(self.circular_mask, getFullKolmogorovCovarianceMatrix(self.nModes))
        kl.generateFromBase(zbase)
        self.modesGenerator = kl
        wf = kl.getKLDict(list(range(self.start_mode, self.start_mode + self.nModes)))

        im = np.zeros((self.nModes, self.user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(self.start_mode, self.start_mode + self.nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=self.user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True, **kwargs)

 
    @returns(ModalCoefficients)
    # write measureModalCoefficientsFromWavefront method here

    def measureModalCoefficientsFromWavefront(
        self,
        wavefront,
        circular_mask,
        user_mask,
        nModes=None,
        start_mode=0,
        **kwargs
    ):
        if nModes is not None:
            self.nModes = nModes
        assert isinstance(wavefront, Wavefront), (
            "wavefront argument must be of type Wavefront, instead is %s"
            % wavefront.__class__.__name__
        )
        assert isinstance(circular_mask, CircularMask), (
            "Circular mask argument must be of type CircularMask, instead is %s"
            % circular_mask.__class__.__name__
        )
        assert isinstance(user_mask, BaseMask), (
            "User mask argument must be of type BaseMask, instead is %s"
            % user_mask.__class__.__name__
        )
        if not np.all(
            circular_mask.as_masked_array() * user_mask.as_masked_array()
            == user_mask.as_masked_array()
        ):
            raise Exception("User mask must be fully contained in circular mask")
        self.base = "KL"
        self.user_mask = user_mask
        self.mask = user_mask
        self.circular_mask = circular_mask
        self.start_mode = start_mode

        if len(kwargs.keys()) == 0:
            self.reconstructor, self.rank = self._synthModeRecFromWavefront()
        else:
            self.reconstructor, self.rank = self._uncachedSynthModeRecFromWavefront(**kwargs)
            
        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        wavefrontInMaskVectorNoPiston = (
            wavefrontInMaskVector - wavefrontInMaskVector.mean()
        )
        return ModalCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, self.reconstructor)
        )


