import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.utils.karhunen_loeve_generator import KarhunenLoeveGenerator as KLGenerator
from arte.utils.rbf_generator import RBFGenerator
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class RadialBasisModalDecomposer(object):

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
        self.nModes = len(coordinates_list)
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

    @cacheResult
    def _synthModeRecFromWavefront(self):

        rbf = RBFGenerator(
            self.circular_mask, self.coordinates_list, rbfFunction=self.base
        )
        rbf.generate()
        self.modesGenerator = rbf
        wf = rbf.getRBFDict(list(range(self.nModes)))

        im = np.zeros((self.nModes, self.user_mask.as_masked_array().compressed().size))
        # modesIdx = list(range(start_mode, start_mode + nModes))
        modesIdx = list(range(self.nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=self.user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True)
        
    def _uncachedSynthModeRecFromWavefront(self, **kwargs):

        rbf = RBFGenerator(
            self.circular_mask, self.coordinates_list, rbfFunction=self.base
        )
        rbf.generate()
        self.modesGenerator = rbf
        wf = rbf.getRBFDict(list(range(self.nModes)))

        im = np.zeros((self.nModes, self.user_mask.as_masked_array().compressed().size))
        # modesIdx = list(range(start_mode, start_mode + nModes))
        modesIdx = list(range(self.nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=self.user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        self.baseModes = im
        return pinv(im, return_rank=True, **kwargs)


    @returns(ModalCoefficients)
    def measureModalCoefficientsFromWavefront(
        self, wavefront, base, circular_mask, user_mask, coordinates_list=None, **kwargs
    ):
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

        if coordinates_list is not None:
            self.coordinates_list = coordinates_list
        self.circular_mask = circular_mask
        self.user_mask = user_mask
        self.mask = user_mask
        self.base = base

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


# write test class for ModalDecomposer here derived from unittest.TestCase
import unittest
import numpy as np
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront


class ModalDecomposerTest(unittest.TestCase):

    def setUp(self):
        self._center = (55, 60)
        self._radius = 32
        self._nPxX = 150
        self._nPxY = 101
        self._nModes = 100
        self._nCoord = self._nModes
        self._mask = CircularMask((self._nPxX, self._nPxY), self._radius, self._center)
        self._user_mask = CircularMask((self._nPxX, self._nPxY), 32, self._center)
        self._wavefront = Wavefront(np.zeros((self._nPxY, self._nPxX)))
        # Example usage:
        _mask = CircularMask((self._nPxY, self._nPxX), self._radius, self._center)
        # Generate coordinates couples

        y0, x0 = self._mask.center()
        cc = np.expand_dims((x0, y0), axis=(1, 2))
        Y, X = (
            np.mgrid[0.5 : self._nPxY + 0.5 : 1, 0.5 : self._nPxX + 0.5 : 1] - cc
        ) / self._mask.radius()
        r = np.sqrt(X**2 + Y**2)
        self._wavefront = Wavefront(r)
        self._user_mask = CircularMask((self._nPxX, self._nPxY), 30, self._center)



    def testRBFModalDecomposer(self):
        self._base = "TPS_RBF"
        self._coords = (
            (70, 60),
            (40, 40),
            (60, 60),
            (60, 80),
            (70, 60 + 2),
            (40 - 2, 40),
            (60 - 3, 60),
            (60, 80 + 3),
            (70 + 5, 60),
            (40, 40 + 5),
        )
        self._modal_decomposer = RadialBasisModalDecomposer(self._coords)

        c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront,
            self._base,
            self._mask,
            self._user_mask,
            coordinates_list=self._coords,
        )

        cTemplate = [-0.598514,  0.578509,  1.123119,  0.561612,  1.730237, -1.918513,
       -2.387599, -0.224815, -0.930808,  1.880122]

        np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)

        tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront,
            self._base,
            self._mask,
            self._user_mask,
            coordinates_list=self._coords,
            rcond=0.995,
        )

        np.testing.assert_allclose(self._modal_decomposer.getRank(), 7)


if __name__ == "__main__":
    unittest.main()
