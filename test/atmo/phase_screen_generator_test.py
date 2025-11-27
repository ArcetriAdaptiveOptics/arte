#!/usr/bin/env python
import unittest
import numpy as np
from arte.atmo.phase_screen_generator import PhaseScreenGenerator
from arte.types.mask import CircularMask
from arte.atmo import von_karman_psd
import astropy.units as u

import os
import tempfile


class PhaseScreenGeneratorTest(unittest.TestCase):

    def setUp(self):
        self._screenSizeInMeters = 8.0
        self._screenSizeInPixels = 128
        self._outerScaleInMeters = 100
        self._r0At500nm = 0.1
        self._psg = PhaseScreenGenerator(self._screenSizeInPixels,
                                         self._screenSizeInMeters,
                                         self._outerScaleInMeters)
        tempdir = tempfile.mkdtemp()
        self.filepath = os.path.join(tempdir, 'test_screens.fits')

    def testGeneratePhaseScreens(self):
        howMany = 2
        wavelengthInMeters = 1e-6
        self._psg.generate_normalized_phase_screens(howMany)
        self._psg.rescale_to(self._r0At500nm)
        phaseScreens = self._psg.get_in_radians_at(wavelengthInMeters)
        expectedShape = (
            howMany, self._screenSizeInPixels, self._screenSizeInPixels)
        self.assertEqual(expectedShape, phaseScreens.shape)


    def test_phase_screens_have_zero_piston(self):
        howMany = 2
        wavelengthInMeters = 1e-6
        self._psg.generate_normalized_phase_screens(howMany)
        self._psg.rescale_to(self._r0At500nm)
        phaseScreens = self._psg.get_in_radians_at(wavelengthInMeters)
        self.assertAlmostEqual(0, phaseScreens[0].mean())
        self.assertAlmostEqual(0, phaseScreens[1].mean())


    def test_save_and_overwrite(self):
        howMany = 2
        self._psg.generate_normalized_phase_screens(howMany)
        self._psg.save_normalized_phase_screens(self.filepath)

        with self.assertRaises(OSError):
            self._psg.save_normalized_phase_screens(self.filepath)

        self._psg.save_normalized_phase_screens(self.filepath, overwrite=True) # should not raise
        


class TestPhaseScreens(unittest.TestCase):

    def setUp(self):
        self.pupil_diameter = 40
        self.r0 = 0.1
        self._lambda = 0.5e-6
        self.L0 = 100
        self._howMany = 10
        self._nPx = 128

    def meanStd(self, ps):
        return np.mean(np.std(ps, axis=(1, 2)))

    def stdInRad(self):
        r0=self.r0AtLambda(self.r0, self._lambda)
        res = von_karman_psd.rms(
            self.pupil_diameter*u.m, self._lambda*u.m, r0*u.m, self.L0*u.m)
        return res.to(u.m) / (self._lambda*u.m) * 2 * np.pi
        #return np.sqrt(1.0299 * (dTele / r0AtLambda)**(5. / 3))

    @staticmethod
    def r0AtLambda(r0At500, wavelenghtInMeters):
        return r0At500 * (wavelenghtInMeters / 0.5e-6) ** (6. / 5)

    def test_kolm_std(self):
        psg = PhaseScreenGenerator(self._nPx, self.pupil_diameter, self.L0)
        psg.generate_normalized_phase_screens(self._howMany)
        mask = CircularMask((self._nPx, self._nPx))
        psg.rescale_to(self.r0)
        got = self.meanStd(np.ma.masked_array(
            psg.get_in_radians_at(self._lambda),
            np.tile(mask.mask(), (self._howMany, 1, 1))))
        want = self.stdInRad()
        print('%g %g %g -> got %g want %g rad  -  ratio %f' %
              (self.pupil_diameter, self.r0, self._lambda, got, want, want / got))
        self.assertAlmostEqual(want, got, delta=0.3 * want)


if __name__ == "__main__":
    unittest.main()
