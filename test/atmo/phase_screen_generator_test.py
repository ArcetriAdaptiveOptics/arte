#!/usr/bin/env python
import unittest
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.types.domainxy import DomainXY
from arte.types.scalar_bidimensional_function import ScalarBidimensionalFunction


class PhaseScreenGeneratorTest(unittest.TestCase):

    def setUp(self):
        self._screenSizeInMeters = 8.0
        self._screenSizeInPixels = 128
        self._outerScaleInMeters = 100
        self._r0At500nm = 0.1
        self._psg = PhaseScreenGenerator(self._screenSizeInPixels,
                                         self._screenSizeInMeters,
                                         self._outerScaleInMeters)

    def testGeneratePhaseScreens(self):
        howMany = 2
        wavelengthInMeters = 1e-6
        self._psg.generateNormalizedPhaseScreens(howMany)
        self._psg.rescaleTo(self._r0At500nm)
        phaseScreens = self._psg.getInRadiansAt(wavelengthInMeters)
        expectedShape = (
            howMany, self._screenSizeInPixels, self._screenSizeInPixels)
        self.assertEqual(expectedShape, phaseScreens.shape)

    def testA(self):
        howMany = 2
        wavelengthInMeters = 1e-6
        self._psg.generateNormalizedPhaseScreens(howMany)
        self._psg.rescaleTo(self._r0At500nm)
        phaseScreens = self._psg.getInRadiansAt(wavelengthInMeters)
        dxy = DomainXY.from_shape(
            phaseScreens[0].shape,
            pixel_size=self._screenSizeInMeters / self._screenSizeInPixels)
        scrF = ScalarBidimensionalFunction(phaseScreens[0], domain=dxy)

if __name__ == "__main__":
    unittest.main()
