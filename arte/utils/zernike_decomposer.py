import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes
from warnings import warn


class ZernikeModalDecomposer(object):

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
        """This throws a deprecation warning on initialization."""
        warn(
            f"{self.__class__.__name__} will be deprecated or change use. Use ZernikeModalDecomposer instead",
            DeprecationWarning,
            stacklevel=2,
        )

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

    @cacheResult
    def _synthZernikeRecFromSlopes(self, nModes, circular_mask, **kwargs):
        zg = ZernikeGenerator(circular_mask)
        dx = zg.getDerivativeXDict(list(range(2, 2 + nModes)))
        dy = zg.getDerivativeYDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, 2 * dx[2].compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            im[i, :] = np.hstack((dx[idx].compressed(), dy[idx].compressed()))
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True, **kwargs)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromSlopes(self, slopes, mask, nModes=None, **kwargs):
        if nModes is None:
            nModes = self.nModes
        assert isinstance(slopes, Slopes), (
            "slopes argument must be of type Slopes, instead is %s"
            % slopes.__class__.__name__
        )
        assert isinstance(mask, CircularMask), (
            "Mask argument must be of type CircularMask, instead is %s"
            % mask.__class__.__name__
        )
        self.mask = mask
        self.reconstructor, self.rank = self._synthZernikeRecFromSlopes(
            nModes, mask, **kwargs
        )

        slopesInMaskVector = np.hstack(
            (
                np.ma.masked_array(slopes.mapX(), mask.mask()).compressed(),
                np.ma.masked_array(slopes.mapY(), mask.mask()).compressed(),
            )
        )

        return ZernikeCoefficients.fromNumpyArray(
            np.dot(slopesInMaskVector, self.reconstructor)
        )

    @cacheResult
    def _synthZernikeRecFromWavefront(self, nModes, circular_mask, user_mask, **kwargs):
        zg = ZernikeGenerator(circular_mask)
        self.modesGenerator = zg
        wf = zg.getZernikeDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True, **kwargs)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, nModes=None, **kwargs
    ):
        self.baseName = "ZERNIKE"
        if nModes is None:
            nModes = self.nModes
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

        self.mask = user_mask
        self.reconstructor, self.rank = self._synthZernikeRecFromWavefront(
            nModes, circular_mask, user_mask, **kwargs
        )

        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        wavefrontInMaskVectorNoPiston = (
            wavefrontInMaskVector - wavefrontInMaskVector.mean()
        )
        return ZernikeCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, self.reconstructor)
        )

    # Write measureKLCoefficientsFromWavefront method here
    @cacheResult
    def _synthModeRecFromWavefront(self):
        zg = ZernikeGenerator(self.circular_mask)
        wf = zg.getZernikeDict(
            list(range(self.start_mode, self.start_mode + self.nModes))
        )
        self.modesGenerator = zg

        im = np.zeros((self.nModes, self.user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(self.start_mode, self.start_mode + self.nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=self.user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True)

    def _uncachedSynthModeRecFromWavefront(self, **kwargs):
        zg = ZernikeGenerator(self.circular_mask)
        wf = zg.getZernikeDict(
            list(range(self.start_mode, self.start_mode + self.nModes))
        )
        self.modesGenerator = zg

        im = np.zeros((self.nModes, self.user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(self.start_mode, self.start_mode + self.nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=self.user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True, **kwargs)

    @returns(ZernikeCoefficients)
    def measureModalCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, nModes=None, start_mode=0, **kwargs
    ):
        self.baseName = "ZERNIKE"
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

        self.start_mode = start_mode
        self.mask = user_mask
        self.user_mask = user_mask
        self.circular_mask = circular_mask
        if len(kwargs.keys()) == 0:
            self.reconstructor, self.rank = self._synthModeRecFromWavefront()
        else:
            self.reconstructor, self.rank = self._uncachedSynthModeRecFromWavefront(
                **kwargs
            )
        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        wavefrontInMaskVectorNoPiston = (
            wavefrontInMaskVector - wavefrontInMaskVector.mean()
        )
        return ModalCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, self.reconstructor)
        )
