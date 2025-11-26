
class MyWhiteNoisePhaseScreenGenerator(PhaseGenerator):
    def _get_power_spectral_density(self, freqMap):
        return np.ones_like(freqMap)
    
    def _get_scaling(self):
        return 1.0

    def get_phase_screens(self):
        return self._phaseScreens


class AbstractPhaseScreenGeneratorTest(unittest.TestCase):

    def setUp(self):    
        self._psg = MyWhiteNoisePhaseScreenGenerator(
            screenSizeInPixels=256,
            screenSizeInMeters=10.0,
            seed=42)

    def test_white_noise(self):    
        self._psg.generate_normalized_phase_screens(numberOfScreens=2)
        ps = self._psg.get_phase_screens()
        #TODO check that the generated phase screen is actually white noise (within some tolerance)

    