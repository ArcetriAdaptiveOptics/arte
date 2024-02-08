import unittest
from arte.photometry import get_normalized_star_spectrum
from arte.photometry.filters import Filters
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from synphot.models import Box1D
from synphot.spectrum import SpectralElement
import numpy as np
from synphot.observation import Observation

class NormalizedStarSpectrumTest(unittest.TestCase):


    def test_vega(self):

        cts_vega_jR= get_normalized_star_spectrum(
            'vega', 12, Filters.JOHNSON_R).integrate(np.arange(1000, 1200)*u.nm)
        cts_vega_bH= get_normalized_star_spectrum(
            'vega', 12, Filters.BESSELL_H).integrate(np.arange(1000, 1200)*u.nm)
        cts_vega_esoR= get_normalized_star_spectrum(
            'vega', 12, Filters.ESO_ETC_R).integrate(np.arange(1000, 1200)*u.nm)

        assert_quantity_allclose(cts_vega_jR, cts_vega_bH)
        assert_quantity_allclose(cts_vega_jR, cts_vega_esoR)

    def test_compare_with_oaalib(self):
        '''
        Computes the photon flux collected by a WFS with bandpass from 0.4 to 1.0 um,
        with an 8m-telescope with total transmission 0.3 (sky+telescope+optics+detector) 
        from a G0V star with mR=19.
        Collected photons [ph/ms]      14.9898
        '''
        spe = get_normalized_star_spectrum('G0V', 19, Filters.BESSELL90_R)
        filt = SpectralElement(Box1D(0.3, 7000, 6000))
        wanted = 14.9898 * u.ph / u.ms
        area = (8*u.m)**2*np.pi/4
        # integrate resulting spectrum
        got_integrate = (spe*filt).integrate(spe.waveset, 'trapezoid') * area
        # use Observation (and convert from ct/s to ph/s)
        obs = Observation(spe, filt)
        got_with_observation = obs.countrate(area=area)*u.ph/u.ct
        print(wanted)
        print(got_integrate.to(u.ph/u.ms))
        print(got_with_observation.to(u.ph/u.ms))
        assert_quantity_allclose(got_integrate, wanted, rtol=5e-2)
        assert_quantity_allclose(got_with_observation, wanted, rtol=5e-2)


    def test_not_vega(self):

        cts_A0V_jR= get_normalized_star_spectrum(
            'A0V', 12, Filters.JOHNSON_R).integrate(np.arange(1000, 1200)*u.nm)
        cts_A0V_bH= get_normalized_star_spectrum(
            'A0V', 12, Filters.BESSELL_H).integrate(np.arange(1000, 1200)*u.nm)

        self.assertGreater(cts_A0V_bH, cts_A0V_jR)


if __name__ == "__main__":
    unittest.main()