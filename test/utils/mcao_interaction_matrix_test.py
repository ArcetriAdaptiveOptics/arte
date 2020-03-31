'''
Created on 31 mar 2020

@author: giuliacarla
'''

import unittest
import numpy as np
from apposto.utils.footprint_geometry import FootprintGeometry
from apposto.utils.mcao_interaction_matrix import MCAOInteractionMatrix
from apposto.types.aperture import CircularOpticalAperture


class TestMCAOInteractionMatrix(unittest.TestCase):

    def testInteractionMatrixValuesOneLayerOneNgs(self):
        ngs_pos = (30, 0)
        target_pos = (0, 0)
        zenith_angle = 0
        telescope_radius = 10
        fov = 50
        altitude = 10e3
        n_modes_sensed = 3
        idx_modes_corrected = [np.array([2, 3, 4])]

        fp = FootprintGeometry()
        fp.set_zenith_angle(zenith_angle)
        fp.setTelescopeRadiusInMeter(telescope_radius)
        fp.setInstrumentFoV(fov)
        fp.setLayerAltitude(altitude)
        fp.addNgs(ngs_pos[0], ngs_pos[1])
        fp.addTarget(target_pos[0], target_pos[1])
        fp.compute()

        metapupil_radius = fp.getMetapupilFootprint()[0].r
        fp_radius = fp.getNgsFootprint()[0].r
        radius_ratio = fp_radius / metapupil_radius
        dm = CircularOpticalAperture(metapupil_radius, [0, 0, 0])

        im = MCAOInteractionMatrix([fp], [dm], n_modes_sensed,
                                   idx_modes_corrected)
        intMat = im.getInteractionMatrixOffTarget()

        h = fp.getNgsFootprint()[0].x
        a = 0
        fromNegro22 = radius_ratio
        fromNegro33 = radius_ratio
        fromNegro44 = radius_ratio**2
        fromNegro24 = 2 * np.sqrt(3) * (
            h * fp_radius) / metapupil_radius**2 * np.cos(a)
        fromNegro34 = 2 * np.sqrt(3) * (
            h * fp_radius) / metapupil_radius**2 * np.sin(a)

        self.assertAlmostEqual(intMat[0][0], fromNegro22, 3)
        self.assertAlmostEqual(intMat[1][1], fromNegro33, 3)
        self.assertAlmostEqual(intMat[2][2], fromNegro44, 2)
        self.assertAlmostEqual(intMat[0][2], fromNegro24, 2)
        self.assertAlmostEqual(intMat[1][2], fromNegro34, 2)

    def testInteractionMatrixShapeOneLayerOneNgs(self):
        ngs1_pos = (30, 0)
        ngs2_pos = (10, 0)
        target_pos = (0, 0)
        zenith_angle = 0
        telescope_radius = 10
        fov = 50
        altitude = 10e3
        n_modes_sensed = 2
        idx_modes_corrected = [np.array([2, 3])]

        fp = FootprintGeometry()
        fp.set_zenith_angle(zenith_angle)
        fp.setTelescopeRadiusInMeter(telescope_radius)
        fp.setInstrumentFoV(fov)
        fp.setLayerAltitude(altitude)
        fp.addNgs(ngs1_pos[0], ngs1_pos[1])
        fp.addNgs(ngs2_pos[0], ngs2_pos[1])
        fp.addTarget(target_pos[0], target_pos[1])
        fp.compute()

        metapupil_radius = fp.getMetapupilFootprint()[0].r
        dm = CircularOpticalAperture(metapupil_radius, [0, 0, 0])

        im = MCAOInteractionMatrix([fp], [dm], n_modes_sensed,
                                   idx_modes_corrected)
        int_mat = im.getInteractionMatrixOffTarget()

        n_ngs = len(fp.getNgsFootprint())
        shape_want = (n_modes_sensed * n_ngs, idx_modes_corrected[0].size)
        shape_got = int_mat.shape
        self.assertEqual(shape_got, shape_want)
