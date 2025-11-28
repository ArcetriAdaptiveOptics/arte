import unittest
import numpy as np
from arte.atmo.abstract_phase_screen_generator import AbstractPhaseScreenGenerator


class MyWhiteNoisePhaseScreenGenerator(AbstractPhaseScreenGenerator):
    def _get_power_spectral_density(self, freqMap):
        return np.ones_like(freqMap)
    
    def _get_scaling(self):
        return 1.0


class AbstractPhaseScreenGeneratorTest(unittest.TestCase):

    def setUp(self):    
        self._psg = MyWhiteNoisePhaseScreenGenerator(
            screenSizeInPixels=160,
            screenSizeInMeters=10.0,
            seed=42)
    
    @staticmethod
    def meanStd(ps):
        return np.mean(np.std(ps, axis=(1, 2)))

    def test_white_noise(self):    
        self._psg.generate_normalized_phase_screens(numberOfScreens=6, scale_amp=1.0)
        ps = self._psg.get_phase_screens()
        want = self._psg._screenSzInPx
        got = self.meanStd(ps)
        self.assertAlmostEqual(want, got, delta=0.1 * want)

    