import unittest
from arte.photometry import get_normalized_star_spectrum
from arte.photometry.filters import Filters
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose


class NormalizedStarSpectrumTest(unittest.TestCase):


    def test_vega(self):

        cts_vega_jR= get_normalized_star_spectrum(
            'vega', 12, Filters.JOHNSON_R).integrate((1000, 1200)*u.nm)
        cts_vega_bH= get_normalized_star_spectrum(
            'vega', 12, Filters.BESSEL_H).integrate((1000, 1200)*u.nm)
        cts_vega_esoR= get_normalized_star_spectrum(
            'vega', 12, Filters.ESO_ETC_R).integrate((1000, 1200)*u.nm)

        assert_quantity_allclose(cts_vega_jR, cts_vega_bH)
        assert_quantity_allclose(cts_vega_jR, cts_vega_esoR)

    def test_not_vega(self):

        cts_A0V_jR= get_normalized_star_spectrum(
            'A0V', 12, Filters.JOHNSON_R).integrate((1000, 1200)*u.nm)
        cts_A0V_bH= get_normalized_star_spectrum(
            'A0V', 12, Filters.BESSEL_H).integrate((1000, 1200)*u.nm)

        self.assertGreater(cts_A0V_bH, cts_A0V_jR)


if __name__ == "__main__":
    unittest.main()