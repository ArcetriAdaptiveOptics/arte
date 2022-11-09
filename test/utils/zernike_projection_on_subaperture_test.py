'''
Created on 9 nov 2022

@author: giuliacarla
'''
import unittest
import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront
from arte.utils.modal_decomposer import ModalDecomposer
from arte.utils.zernike_projection_on_subaperture import \
    ZernikeProjectionOnSubaperture


class TestZernikeProjectionOnSubaperture(unittest.TestCase):

    def testOnAxisSubPupil(self):
        mpup_radius = 20
        spup_radius = 10
        ratio = spup_radius / mpup_radius
        rho = 0
        theta = 0
        zp = ZernikeProjectionOnSubaperture(
            mpup_radius, spup_radius, rho, theta)
        negro_matrix = zp.getProjectionMatrix()
        got = np.diagonal(negro_matrix)
        want = np.array([
            ratio, ratio, ratio**2, ratio**2, ratio**2,
            ratio**3, ratio**3, ratio**3, ratio**3, ratio**4])
        np.testing.assert_equal(got, want)
    

    def testComparisonWithModalDecomposition(self):
        j = 11
        dx = 5
        dy = 3
        mpup_radius = 26
        zg = ZernikeGenerator(mpup_radius * 2)
        zern = zg.getZernike(index=j)
        spup_radius = 20
        spup_center = zg.center() + [dy, dx]
        spup = CircularMask(
            frameShape = (mpup_radius * 2, mpup_radius * 2),
            maskRadius=spup_radius, maskCenter=spup_center)
        spup_mask = spup.as_masked_array()
        idcs_spup_masked = np.argwhere(spup_mask.flatten().mask == False)
        spup_mask_new = spup_mask.flatten()
        spup_mask_new[idcs_spup_masked] = zern.flatten()[idcs_spup_masked]
        spup_mask_new = spup_mask_new.reshape(spup_mask.shape)
        
        spup_wf = Wavefront(spup_mask_new)
        spup_circ_mask = CircularMask.fromMaskedArray(spup_mask_new)
        md = ModalDecomposer(n_zernike_modes=j-1)
        spup_zern = md.measureZernikeCoefficientsFromWavefront(
            wavefront=spup_wf, mask=spup_circ_mask).toNumpyArray()
            
        y, x = spup_center - zg.center()
        rho = np.sqrt(x**2 + y**2)
        theta_deg = np.rad2deg(np.arctan2(y, x)) 
        zp = ZernikeProjectionOnSubaperture(
            mpup_radius, spup_radius, rho, theta_deg)
        negro_matrix = zp.getProjectionMatrix()[j-2, :j-1]
        np.testing.assert_almost_equal(spup_zern, negro_matrix, 2)
