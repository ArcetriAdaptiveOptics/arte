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
        a = np.arctan2(fp.getNgsFootprint()[0].y, fp.getNgsFootprint()[0].x)
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
        im_off = im.getInteractionMatrixOffTarget()

        n_ngs = len(fp.getNgsFootprint())
        shape_want = (n_modes_sensed * n_ngs, idx_modes_corrected[0].size)
        shape_got = im_off.shape
        self.assertEqual(shape_got, shape_want)

    def testOffTargetInteractionMatrixShape(self):
        ngs1_pos = (30, 0)
        ngs2_pos = (10, 0)
        target_pos = (0, 0)
        zenith_angle = 0
        telescope_radius = 10
        fov = 50
        altitude1 = 0
        altitude2 = 10e3
        n_modes_sensed = 2
        idx_modes_corrected = [np.array([2, 3]), np.array([4])]

        fp1 = FootprintGeometry()
        fp1.set_zenith_angle(zenith_angle)
        fp1.setTelescopeRadiusInMeter(telescope_radius)
        fp1.setInstrumentFoV(fov)
        fp1.setLayerAltitude(altitude1)
        fp1.addNgs(ngs1_pos[0], ngs1_pos[1])
        fp1.addNgs(ngs2_pos[0], ngs2_pos[1])
        fp1.addTarget(target_pos[0], target_pos[1])
        fp1.compute()

        fp2 = FootprintGeometry()
        fp2.set_zenith_angle(zenith_angle)
        fp2.setTelescopeRadiusInMeter(telescope_radius)
        fp2.setInstrumentFoV(fov)
        fp2.setLayerAltitude(altitude2)
        fp2.addNgs(ngs1_pos[0], ngs1_pos[1])
        fp2.addNgs(ngs2_pos[0], ngs2_pos[1])
        fp2.addTarget(target_pos[0], target_pos[1])
        fp2.compute()

        mp_radius_1 = fp1.getMetapupilFootprint()[0].r
        mp_radius_2 = fp2.getMetapupilFootprint()[0].r
        dm1 = CircularOpticalAperture(mp_radius_1, [0, 0, 0])
        dm2 = CircularOpticalAperture(mp_radius_2, [0, 0, 0])

        im = MCAOInteractionMatrix([fp1, fp2], [dm1, dm2], n_modes_sensed,
                                   idx_modes_corrected)
        im_off = im.getInteractionMatrixOffTarget()

        n_ngs = len(fp1.getNgsFootprint())
        shape_want = (n_modes_sensed * n_ngs,
                      len(idx_modes_corrected[0]) + len(idx_modes_corrected[1])
                      )
        shape_got = im_off.shape
        self.assertEqual(shape_got, shape_want)

    def testOnTargetInteractionMatrixValues(self):
        ngs_pos = (30, 0)
        target_pos = (5, 10)
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
        dm = CircularOpticalAperture(metapupil_radius, [0, 0, 0])
        fp_radius = fp.getTargetFootprint()[0].r
        radius_ratio = fp_radius / metapupil_radius

        im = MCAOInteractionMatrix([fp], [dm], n_modes_sensed,
                                   idx_modes_corrected)
        im_on = im.getInteractionMatrixOnTarget()

        h = fp.getTargetFootprint()[0].x
        a = np.arctan2(fp.getTargetFootprint()[0].y,
                       fp.getTargetFootprint()[0].x)
        fromNegro22 = radius_ratio
        fromNegro33 = radius_ratio
        fromNegro44 = radius_ratio**2
        fromNegro24 = 2 * np.sqrt(3) * (
            h * fp_radius) / metapupil_radius**2 * np.cos(a)
        fromNegro34 = 2 * np.sqrt(3) * (
            h * fp_radius) / metapupil_radius**2 * np.sin(a)

        self.assertAlmostEqual(im_on[0][0], fromNegro22, 3)
        self.assertAlmostEqual(im_on[1][1], fromNegro33, 3)
        self.assertAlmostEqual(im_on[2][2], fromNegro44, 2)
        self.assertAlmostEqual(im_on[0][2], fromNegro24, 2)
        self.assertAlmostEqual(im_on[1][2], fromNegro34, 2)

    def testOnTargetInteractionMatrixShape(self):
        ngs1_pos = (30, 0)
        ngs2_pos = (10, 20)
        target_pos = (0, 0)
        zenith_angle = 0
        telescope_radius = 10
        fov = 50
        altitude1 = 0
        altitude2 = 10e3
        n_modes_sensed = 2
        idx_modes_corrected = [np.array([2, 3]), np.array([4])]

        fp1 = FootprintGeometry()
        fp1.set_zenith_angle(zenith_angle)
        fp1.setTelescopeRadiusInMeter(telescope_radius)
        fp1.setInstrumentFoV(fov)
        fp1.setLayerAltitude(altitude1)
        fp1.addNgs(ngs1_pos[0], ngs1_pos[1])
        fp1.addNgs(ngs2_pos[0], ngs2_pos[1])
        fp1.addTarget(target_pos[0], target_pos[1])
        fp1.compute()

        fp2 = FootprintGeometry()
        fp2.set_zenith_angle(zenith_angle)
        fp2.setTelescopeRadiusInMeter(telescope_radius)
        fp2.setInstrumentFoV(fov)
        fp2.setLayerAltitude(altitude2)
        fp2.addNgs(ngs1_pos[0], ngs1_pos[1])
        fp2.addNgs(ngs2_pos[0], ngs2_pos[1])
        fp2.addTarget(target_pos[0], target_pos[1])
        fp2.compute()

        mp_radius_1 = fp1.getMetapupilFootprint()[0].r
        mp_radius_2 = fp2.getMetapupilFootprint()[0].r
        dm1 = CircularOpticalAperture(mp_radius_1, [0, 0, 0])
        dm2 = CircularOpticalAperture(mp_radius_2, [0, 0, 0])

        im = MCAOInteractionMatrix([fp1, fp2], [dm1, dm2], n_modes_sensed,
                                   idx_modes_corrected)
        im_on = im.getInteractionMatrixOnTarget()

        shape_want = (n_modes_sensed,
                      len(idx_modes_corrected[0]) + len(idx_modes_corrected[1])
                      )
        shape_got = im_on.shape
        self.assertEqual(shape_got, shape_want)
