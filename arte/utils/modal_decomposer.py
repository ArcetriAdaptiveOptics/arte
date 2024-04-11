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

    #write class documentation here
    '''
    This class provides methods to decompose a wavefront into a set of modal coefficients.
    The modal coefficients can be Zernike or Karhunen-Loeve (KL) coefficients. The class
    provides methods to measure the modal coefficients from slopes or wavefronts. The
    wavefront can be decomposed into Zernike or KL coefficients. The class also provides
    methods to measure the modal coefficients from wavefronts using a set of radial basis
    functions (RBF) or thin plate splines (TPS) or Gaussian radial basis functions (RBF)
    or inverse quadratic radial basis functions (RBF) or multiquadric radial basis functions
    (RBF) or inverse multiquadric radial basis functions (RBF). The class also provides
    methods to measure the modal coefficients from wavefronts using a set of user defined
    basis functions.
    '''


    def __init__(self, n_modes):
        self.nModes = n_modes

    @cacheResult
    def _synthZernikeRecFromSlopes(self, nModes, circular_mask):
        zg = ZernikeGenerator(circular_mask)
        dx = zg.getDerivativeXDict(list(range(2, 2 + nModes)))
        dy = zg.getDerivativeYDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, 2 * dx[2].compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            im[i, :] = np.hstack((dx[idx].compressed(), dy[idx].compressed()))
            i += 1
        return pinv(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromSlopes(self, slopes, mask, nModes=None):
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

        reconstructor = self._synthZernikeRecFromSlopes(nModes, mask)

        slopesInMaskVector = np.hstack(
            (
                np.ma.masked_array(slopes.mapX(), mask.mask()).compressed(),
                np.ma.masked_array(slopes.mapY(), mask.mask()).compressed(),
            )
        )

        return ZernikeCoefficients.fromNumpyArray(
            np.dot(slopesInMaskVector, reconstructor)
        )

    @cacheResult
    def _synthZernikeRecFromWavefront(self, nModes, circular_mask, user_mask):
        zg = ZernikeGenerator(circular_mask)
        wf = zg.getZernikeDict(list(range(2, 2 + nModes)))
        im = np.zeros((nModes, user_mask.as_masked_array().compressed().size))
        modesIdx = list(range(2, 2 + nModes))

        i = 0
        for idx in modesIdx:
            wf_masked = np.ma.masked_array(wf[idx].data, mask=user_mask.mask())
            im[i, :] = wf_masked.compressed()
            i += 1
        return pinv(im)

    @returns(ZernikeCoefficients)
    def measureZernikeCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, nModes=None
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

        reconstructor = self._synthZernikeRecFromWavefront(
            nModes, circular_mask, user_mask
        )
        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        wavefrontInMaskVectorNoPiston = (
            wavefrontInMaskVector - wavefrontInMaskVector.mean()
        )
        return ZernikeCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, reconstructor)
        )


    #Write measureKLCoefficientsFromWavefront method here
    #@cacheResult
    def _synthModeRecFromWavefront(self, base, nModes, circular_mask, user_mask, start_mode=0, coordinates_list=None):
        if base == "ZERNIKE":
            zg = ZernikeGenerator(circular_mask)
            wf = zg.getZernikeDict(list(range(start_mode, start_mode + nModes)))
        elif base == "KL":
            zz = ZernikeGenerator(circular_mask)
            zbase = np.rollaxis(np.ma.masked_array([zz.getZernike(n) 
                                        for n in range(2,nModes+2)]),0,3)
            kl = KLGenerator(circular_mask, getFullKolmogorovCovarianceMatrix(nModes))
            kl.generateFromBase(zbase)
            wf = kl.getKLDict(list(range(start_mode, start_mode + nModes)))
        elif base == "TPS_RBF" or base == "GAUSS_RBF" or base == "INV_QUADRATIC" or base == "MULTIQUADRIC" or base == "INV_MULTIQUADRIC":
            rbf = RBFGenerator(circular_mask, coordinates_list, rbfFunction=base)
            rbf.generate()
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
        return pinv(im)

    @returns(ModalCoefficients)
    #write measureModalCoefficientsFromWavefront method here

    def measureModalCoefficientsFromWavefront(
        self, wavefront, circular_mask, user_mask, base, nModes=None, start_mode=0, coordinates_list=None
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

        reconstructor = self._synthModeRecFromWavefront(
            base, nModes, circular_mask, user_mask, start_mode, coordinates_list
        )
        wavefrontInMaskVector = np.ma.masked_array(
            wavefront.toNumpyArray(), user_mask.mask()
        ).compressed()
        wavefrontInMaskVectorNoPiston = (
            wavefrontInMaskVector - wavefrontInMaskVector.mean()
        )
        return ModalCoefficients.fromNumpyArray(
            np.dot(wavefrontInMaskVectorNoPiston, reconstructor)
        )    

        
#write test class for ModalDecomposer here derived from unittest.TestCase
import unittest
import numpy as np
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront
       



if __name__ == "__main__":
    unittest.main()



