
import unittest
import numpy as np
import astropy.units as u

from arte.astro.telescopes import VLT
from arte.astro.mag_estimator import MagEstimator


class MagEstimatorTest(unittest.TestCase):

    def test_eris(self):
        '''
        Use some known ERIS numbers to test magnitude.
        This is the LO channel in TN 20200203_173333.
        '''
        est = MagEstimator(total_adus = 65675.0 * u.adu,
                           detector_gain = 64,
                           detector_adu_e_ratio = 16.4 * u.electron / u.adu,
                           detector_nsubaps = 16,
                           detector_freq = 498 * u.Hz,
                           wfs_transmission = 0.42,
                           detector_bandname = 'R',
                           detector_bandwidth = 300 * u.nm,
                           detector_qe = 1 * u.electron / u.ph,
                           telescope = VLT)

        assert np.isclose(est.mag(), 11.885415577)

if __name__ == "__main__":
    unittest.main()

# ___oOo___
