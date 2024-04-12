import numpy as np
from scipy.linalg import pinv
from arte.utils.decorator import cacheResult, returns
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.zernike_coefficients import ZernikeCoefficients
from arte.atmo.utils import getFullKolmogorovCovarianceMatrix
from arte.utils.kl_generator import KLGenerator
from arte.utils.rbf_generator import RBFGenerator
from arte.types.modal_coefficients import ModalCoefficients
from arte.types.mask import CircularMask, BaseMask
from arte.types.wavefront import Wavefront
from arte.types.slopes import Slopes


class ModalDecomposer(object):

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

    def __init__(self, n_modes):
        self.nModes = n_modes
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
    # @cacheResult
    def _synthModeRecFromWavefront(
        self,
        base,
        nModes,
        circular_mask,
        user_mask,
        start_mode=0,
        coordinates_list=None,
        **kwargs
    ):
        self.baseName = base
        if base == "ZERNIKE":
            zg = ZernikeGenerator(circular_mask)
            wf = zg.getZernikeDict(list(range(start_mode, start_mode + nModes)))
            self.modesGenerator = zg
        elif base == "KL":
            zz = ZernikeGenerator(circular_mask)
            zbase = np.rollaxis(
                np.ma.masked_array([zz.getZernike(n) for n in range(2, nModes + 2)]),
                0,
                3,
            )
            kl = KLGenerator(circular_mask, getFullKolmogorovCovarianceMatrix(nModes))
            kl.generateFromBase(zbase)
            self.modesGenerator = kl
            wf = kl.getKLDict(list(range(start_mode, start_mode + nModes)))
        elif (
            base == "TPS_RBF"
            or base == "GAUSS_RBF"
            or base == "INV_QUADRATIC"
            or base == "MULTIQUADRIC"
            or base == "INV_MULTIQUADRIC"
        ):
            rbf = RBFGenerator(circular_mask, coordinates_list, rbfFunction=base)
            rbf.generate()
            self.modesGenerator = rbf
            wf = rbf.getRBFDict(list(range(start_mode, start_mode + nModes)))
        else:
            raise ValueError("Invalid base %s" % base)

        im = np.zeros((nModes, user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(start_mode, start_mode + nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
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
        base,
        nModes=None,
        start_mode=0,
        coordinates_list=None,
        **kwargs
    ):
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
        self.reconstructor, self.rank = self._synthModeRecFromWavefront(
            base,
            nModes,
            circular_mask,
            user_mask,
            start_mode,
            coordinates_list,
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


# write test class for ModalDecomposer here derived from unittest.TestCase
import unittest
import numpy as np
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront


# write test class for ModalDecomposer here derived from unittest.TestCase which tests the measureModalCoefficientsFromWavefront method
# and save the modal coefficients in a variable and compare it with the expected modal coefficients
class TestModalDecomposer(unittest.TestCase):

    def setUp(self):
        self._nModes = 10
        self._nPxX = 150
        self._nPxY = 101
        self._mask = CircularMask((self._nPxX, self._nPxY), 30, (55, 60))

        y0, x0 = self._mask.center()
        cc = np.expand_dims((x0, y0), axis=(1, 2))
        Y, X = (
            np.mgrid[0.5 : self._nPxY + 0.5 : 1, 0.5 : self._nPxX + 0.5 : 1] - cc
        ) / self._mask.radius()
        r = np.sqrt(X**2 + Y**2)
        self._wavefront = Wavefront(r)
        self._user_mask = CircularMask((self._nPxX, self._nPxY), 30, (55, 60))

    def testZernikeModalDecomposer(self):
        self._base = "ZERNIKE"
        self._modal_decomposer = ModalDecomposer(self._nModes)
        c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront, self._mask, self._user_mask, self._base, start_mode=1
        )
        cTemplate = [
            -5.57006321e-05,
            1.08810731e-01,
            -1.08810731e-01,
            2.15396089e-01,
            1.08517924e-02,
            -1.24900090e-16,
            2.82709210e-02,
            -2.82709210e-02,
            -1.08170522e-03,
            -1.08170522e-03,
        ]
        cTemplate = [-1.01297584e-05, -1.14291429e-03, -1.81305804e-01,  3.91720929e-02,
            3.52736742e-02, -2.25641006e-02,  5.48724126e-03, -1.86942642e-03,
            -4.50770580e-03, -3.67539262e-03]
        np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)
        tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            self._wavefront, self._mask, self._user_mask, self._base, start_mode=1, rtol=0.995)
        np.testing.assert_allclose(self._modal_decomposer.getRank(), 8)

    #def testKLModalDecomposer(self):
        #self._base = "KL"
        #self._modal_decomposer = ModalDecomposer(self._nModes)
        #c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            #self._wavefront, self._mask, self._user_mask, self._base
        #)
        #cTemplate = [-1.09658355e-01, -1.09658355e-01, -2.17513529e-01, -9.71445147e-17,
            #1.08517924e-02, -1.08170522e-03, -1.08170522e-03, -2.47803383e-02,
            #2.47803383e-02,  7.58052411e-03]
        #np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)
        #tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            #self._wavefront, self._mask, self._user_mask, self._base, rtol=0.995)
        #np.testing.assert_allclose(self._modal_decomposer.getRank(), 8)


    #def testRBFModalDecomposer(self):
        #self._base = "TPS_RBF"
        #self._nCoords = 3
        #self._coords = ((70,60),(40,40), (60,60), (60,80), (70,60+2),(40-2,40), (60-3,60), (60,80+3), (70+5,60),(40,40+5))
        #self._modal_decomposer = ModalDecomposer(self._nModes)
        #c2test = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            #self._wavefront, self._mask, self._user_mask, self._base, coordinates_list=self._coords
        #)
        #cTemplate = [7.76699114,  1.0759681,  -4.90487223,  2.39658433, -3.70662821, -1.74705926,
  #2.28727819, -1.5312593,  -1.87877108,  1.36101628]
        #np.testing.assert_allclose(c2test.toNumpyArray(), cTemplate, rtol=1e-5)
        #tmp = self._modal_decomposer.measureModalCoefficientsFromWavefront(
            #self._wavefront, self._mask, self._user_mask, self._base, coordinates_list=self._coords, rtol=0.995)
        #np.testing.assert_allclose(self._modal_decomposer.getRank(), 1)



if __name__ == "__main__":
    unittest.main()
