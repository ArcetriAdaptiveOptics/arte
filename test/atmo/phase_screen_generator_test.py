#!/usr/bin/env python
import unittest
import numpy as np
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.types.domainxy import DomainXY
from arte.types.scalar_bidimensional_function import ScalarBidimensionalFunction
from arte.types.mask import CircularMask


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
        self._psg.generate_normalized_phase_screens(howMany)
        self._psg.rescale_to(self._r0At500nm)
        phaseScreens = self._psg.get_in_radians_at(wavelengthInMeters)
        expectedShape = (
            howMany, self._screenSizeInPixels, self._screenSizeInPixels)
        self.assertEqual(expectedShape, phaseScreens.shape)

    def testA(self):
        howMany = 2
        wavelengthInMeters = 1e-6
        self._psg.generate_normalized_phase_screens(howMany)
        self._psg.rescale_to(self._r0At500nm)
        phaseScreens = self._psg.get_in_radians_at(wavelengthInMeters)
        dxy = DomainXY.from_shape(
            phaseScreens[0].shape,
            pixel_size=self._screenSizeInMeters / self._screenSizeInPixels)
        scrF = ScalarBidimensionalFunction(phaseScreens[0], domain=dxy)


class TestPhaseScreens(unittest.TestCase):

    def setUp(self):
        self._dPup = 8
        self._r0 = 0.2
        self._lambda = 0.5e-6
        self._L0 = 1e9
        self._howMany = 100
        self._nPx = 128

    def meanStd(self, ps):
        return np.mean(np.std(ps, axis=(1, 2)))

    def stdInRad(self, dTele, r0AtLambda):
        return np.sqrt(1.0299 * (dTele / r0AtLambda)**(5. / 3))

    @staticmethod
    def r0AtLambda(r0At500, wavelenghtInMeters):
        return r0At500 * (wavelenghtInMeters / 0.5e-6) ** (6. / 5)

    def test(self):
        psg = PhaseScreenGenerator(self._nPx, self._dPup, self._L0)
        psg.generate_normalized_phase_screens(self._howMany)
        mask = CircularMask((self._nPx, self._nPx))
        psg.rescale_to(self._r0)
        got = self.meanStd(np.ma.masked_array(
            psg.get_in_radians_at(self._lambda),
            np.tile(mask.mask(), (self._howMany, 1, 1))))
        want = self.stdInRad(self._dPup,
                             self.r0AtLambda(self._r0, self._lambda))
        print('%g %g %g -> got %g want %g rad  -  ratio %f' %
              (self._dPup, self._r0, self._lambda, got, want, want / got))
        self.assertAlmostEqual(want, got, delta=0.3 * want)


if __name__ == "__main__":
    unittest.main()
