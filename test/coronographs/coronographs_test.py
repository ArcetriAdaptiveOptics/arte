import unittest
import numpy as np
from arte.utils.decorator import override


from arte.optical_propagation.abstract_coronograph import Coronograph
from arte.optical_propagation.lyot_coronograph import LyotCoronograph, KnifeEdgeCoronograph
from arte.optical_propagation.four_quadrant_coronograph import FourQuadrantCoronograph
from arte.optical_propagation.vortex_coronograph import VortexCoronograph
from arte.optical_propagation.perfect_coronograph import PerfectCoronograph


class MyCoronographTest(Coronograph):
    """ Design a coronograph that perfectly 
    cancels the phase and multiplies the 
    amplitude by a scalar """

    def __init__(self, scaleVal:float):
        self._apodizationValue = scaleVal

    @override
    def _get_focal_plane_mask(self, field):
        return np.conj(field)

    @override
    def _get_pupil_mask(self, field):
        return np.ones_like(field)
    
    @override
    def _get_apodizer(self):
        return self._apodizationValue


class AbstractCoronographTest(unittest.TestCase):

    def setUp(self):
        scaleVal = 3.0    
        self._coro = MyCoronographTest(scaleVal=scaleVal)


    def test_white_noise(self):    
        Npix = 32
        oversampling = 2
        psfExpectedShape = (Npix*oversampling,Npix*oversampling)
        fieldAmp = np.ones([Npix,Npix])
        field = fieldAmp * np.exp(1j*np.random.randn([Npix,Npix]),dtype=np.complex128)

        psf = self._coro.get_coronographic_psf(field, oversampling=oversampling)
        self.assertEqual(psf.shape, psfExpectedShape) # check the shape

    