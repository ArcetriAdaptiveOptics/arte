#!/usr/bin/env python
import unittest
from apposto.misc.fourier_adaptive_optics import FourierAdaptiveOptics

__version__ = "$Id:$"


class FourierAdaptiveOpticsTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def testConstruction(self):
        pupDiameter= 16.0
        wavelength= 1.35e-6
        focalPlaneFoV= 0.8
        resolutionFactor= 3.0
        fao= FourierAdaptiveOptics(pupDiameter,
                                   wavelength,
                                   focalPlaneFoV,
                                   resolutionFactor)
        self.assertEqual(pupDiameter, fao.pupilDiameterInMeters())
        self.assertEqual(wavelength, fao.wavelengthInMeters())
        self.assertEqual(focalPlaneFoV, fao.focalPlaneFieldOfViewInArcsec())
        self.assertEqual(resolutionFactor, fao.resolutionFactor())


    def testPixelScales(self):
        fao= FourierAdaptiveOptics(8.0, 1e-6, 1.290, 4.0)
        self._checkPixelScale(fao)
        fao= FourierAdaptiveOptics(42.0, 2.2e-6, 1.00, 2.0)
        self._checkPixelScale(fao)
        fao= FourierAdaptiveOptics(1.0, 0.5e-6, 10.00, 5.2)
        self._checkPixelScale(fao)


    def _checkPixelScale(self, fao):
        wavelength= fao.wavelengthInMeters()
        pupDiameter= fao.pupilDiameterInMeters()
        fov= fao.focalPlaneFieldOfViewInArcsec()
        resolutionFactor= fao.resolutionFactor()

        wantedPixelScale= (0.5 * wavelength / pupDiameter / resolutionFactor *
                           FourierAdaptiveOptics.RAD2ARCSEC)
        wantedFocalPlaneSizeInPixels= int(fov / wantedPixelScale)
        wantedAngularFreqPxSz= 1 / (
            wantedFocalPlaneSizeInPixels * wantedPixelScale *
            FourierAdaptiveOptics.ARCSEC2RAD)
        wantedPupilPlaneSpatialFreqPxSz= (
            wantedPixelScale * FourierAdaptiveOptics.ARCSEC2RAD /
            wavelength)
        wantedPupilPlanePxSz= 1 / (
            wantedFocalPlaneSizeInPixels * wantedPupilPlaneSpatialFreqPxSz)

        self.assertAlmostEqual(wantedPixelScale,
                               fao.focalPlanePixelSizeInArcsec())
        self.assertAlmostEqual(wantedFocalPlaneSizeInPixels,
                               fao.focalPlaneSizeInPixels())
        self.assertAlmostEqual(
            wantedAngularFreqPxSz,
            fao.focalPlaneAngularFrequencyPixelSizeInInverseRadians())
        self.assertAlmostEqual(
            wantedPupilPlaneSpatialFreqPxSz,
            fao.pupilPlaneSpatialFrequencyPizelSizeInInverseMeters())
        self.assertAlmostEqual(
            wantedPupilPlanePxSz,
            fao.pupilPlanePixelSizeInMeters())

        self._logPixelScales(fao)

    def _logPixelScales(self, fao):
        print('Pupil Diameter [m] %g' % fao.pupilDiameterInMeters())
        print('Wavelength [m] %g' % fao.wavelengthInMeters())
        print('Focal Plane FoV [arcsec] %g' %
              fao.focalPlaneFieldOfViewInArcsec())
        print('Focal Plane size [px] %g' % fao.focalPlaneSizeInPixels())
        print('Focal Plane Pixel Size [arcsec] %g' %
              fao.focalPlanePixelSizeInArcsec())
        print('Focal Plane Angular Frequency Pixel Size [1/rad] %g' %
              fao.focalPlaneAngularFrequencyPixelSizeInInverseRadians())
        print('Pupil Plane Pixel Size [m] %g' %
              fao.pupilPlanePixelSizeInMeters())
        print('Pupil Plane Spatial Frequency Pixel Size [1/m] %g' %
              fao.pupilPlaneSpatialFrequencyPizelSizeInInverseMeters())


    def testFocalPlaneAngularFrequencyCoordinates(self):
        fao= FourierAdaptiveOptics(8.0, 1e-6, 1.290, 4.0)
        coords= fao.focalPlaneAngularFrequencyCoordinatesInInverseRadians()
         
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()