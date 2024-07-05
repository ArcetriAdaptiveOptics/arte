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

    def generator(self, nModes, circular_mask, user_mask, rbfFunction=None, **kwargs):
        rbf = RBFGenerator(circular_mask, self.coordinates_list, rbfFunction=rbfFunction)
        rbf.generate()
        return rbf


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
            self._mask,
            self._user_mask,
            rbfFunction=self._base,
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
