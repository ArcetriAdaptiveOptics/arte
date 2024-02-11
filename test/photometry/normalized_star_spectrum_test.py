import os
import unittest
from arte.photometry import get_normalized_star_spectrum
from arte.photometry.filters import Filters
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
import synphot
from synphot.models import Box1D
from synphot.spectrum import SpectralElement, SourceSpectrum
import numpy as np
from synphot.observation import Observation

from arte.utils.package_data import dataRootDir
from astropy.io import fits

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

    def test_can_integrate_full_range(self):
        filt = Filters.get(Filters.BESSELL_H)
        spec = get_normalized_star_spectrum('vega', 0, Filters.BESSELL_H)
        ct_1400_1900 = (spec*filt).integrate(np.arange(1400, 1900)*u.nm)
        ct_100_5000 = (spec*filt).integrate(np.arange(100, 5000)*u.nm)
        assert_quantity_allclose(ct_1400_1900, ct_100_5000)

    def test_oaalib_compare_wit(self):
        '''
        Replicate case below and check results agree within 5%
        Use synphot Observation or integrate
        
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

    def test_oaalib_compare_vega_spectra(self):
        import matplotlib.pyplot as plt

        def _load_oaalib_vega():
            rootDir = dataRootDir()
            fname = os.path.join(rootDir, 'photometry',
                                 'spectra', 'vega_k93_array.fits')
            dd = fits.getdata(fname)
            return dd[0]*u.AA, dd[1]*u.erg / (u.s * u.cm**2 * u.AA)

        oaa_l, oaa_s = _load_oaalib_vega()

        spv = SourceSpectrum.from_vega()
        spv_erg = synphot.units.convert_flux(
            oaa_l, spv(oaa_l), u.erg / (u.s * u.cm**2 * u.AA))

        def _plot():
            plt.plot(oaa_l, (oaa_s-spv_erg)/spv_erg)
            plt.xlabel('Angstrom')
            plt.ylabel('vega spectrum: (OAA-SYNPHOT)/SYNPHOT')
            plt.grid()
            plt.semilogx()
            plt.xlim(4500, 22000)
            plt.ylim(-0.1, 0.1)

        def _range(wl_from, wl_to):
            return np.flatnonzero((oaa_l > wl_from) & (oaa_l < wl_to))

        # assert spectra are equal within 2% when integrated over R J H bands
        limr = _range(600*u.nm, 1000*u.nm)
        assert_quantity_allclose(np.trapz(oaa_s[limr], x=oaa_l[limr]),
                                 np.trapz(spv_erg[limr], x=oaa_l[limr]), rtol=0.02)

        limr = _range(1100*u.nm, 1400*u.nm)
        assert_quantity_allclose(np.trapz(oaa_s[limr], x=oaa_l[limr]),
                                 np.trapz(spv_erg[limr], x=oaa_l[limr]), rtol=0.02)

        limr = _range(1500*u.nm, 1800*u.nm)
        assert_quantity_allclose(np.trapz(oaa_s[limr], x=oaa_l[limr]),
                                 np.trapz(spv_erg[limr], x=oaa_l[limr]), rtol=0.02)

    def test_oaalib_compare_normalized(self):
        import matplotlib.pyplot as plt

        def _load_oaalib_K5V_R():
            rootDir = dataRootDir()
            fname = os.path.join(rootDir, 'photometry',
                                 'spectra', 'sp_K5V_norm_R0_ESO.fits')
            dd = fits.getdata(fname)
            return dd[0]*u.um, dd[1]*u.ph / (u.s * u.m**2 * u.um)

        # load oaalib K5V normalized to 0 mag on ESO_ETC_R
        oaa_l, oaa_s = _load_oaalib_K5V_R()

        # get normalized spectrum
        spv = get_normalized_star_spectrum('K5V', 0, Filters.ESO_ETC_R)
        # convert to oaa lib units
        spv_erg = synphot.units.convert_flux(
            oaa_l, spv(oaa_l), u.ph / (u.s * u.m**2 * u.um))

        def _plot():
            plt.plot(oaa_l, (oaa_s-spv_erg)/spv_erg)
            plt.xlabel('Micron')
            plt.ylabel('vega spectrum: (OAA-SYNPHOT)/SYNPHOT')
            plt.grid()
            plt.semilogx()
            plt.xlim(0.4500, 2.2000)
            plt.ylim(-0.1, 0.1)

        def _range(wl_from, wl_to):
            return np.flatnonzero((oaa_l > wl_from) & (oaa_l < wl_to))

        # compute fluxes in 1500-1800nm
        limr = _range(1500*u.nm, 1800*u.nm)
        oaa_flux = np.trapz(oaa_s[limr], x=oaa_l[limr])
        arte_flux = np.trapz(spv_erg[limr], x=oaa_l[limr])
        print("oaa_lib %s / arte %s  - diff %g %%" %
              (oaa_flux, arte_flux, (arte_flux-oaa_flux)/oaa_flux*100))

        # assert spectra are equal within 2% when integrated over R J H bands
        assert_quantity_allclose(oaa_flux, arte_flux, rtol=0.02)

    def test_countrate_of_normalized_spectra_on_Bessel(self):
        '''
        Check integrated flux of normalized spectra on a given filter is the same
        
        Assert countrate for K5V and A0V normalized on the filter Bessel90_R 
        is identical to the one for vega within 0.1%
        '''
        self._test_countrate_normalized(Filters.BESSELL90_R)

    def test_countrate_of_normalized_spectra_on_ESO_ETC_R(self):
        '''
        Check integrated flux of normalized spectra on a given filter is the same
        
        Assert countrate for K5V and A0V normalized on the filter ESO_ETC_R 
        is identical to the one for vega within 0.1%
        '''
        self._test_countrate_normalized(Filters.ESO_ETC_R)

    def _test_countrate_normalized(self, filtername):
        f_r = Filters.get(filtername)
        sp_vega = get_normalized_star_spectrum('vega', 0, filtername)
        sp_k5V = get_normalized_star_spectrum('K5V', 0, filtername)
        sp_a0V = get_normalized_star_spectrum('A0V', 0, filtername)

        cts_k5V_i = (sp_k5V*f_r).integrate(f_r.waveset)
        cts_a0V_i = (sp_a0V*f_r).integrate(f_r.waveset)
        cts_vega_i = (sp_vega*f_r).integrate(f_r.waveset)
        cts_k5V = Observation(sp_k5V, f_r).countrate(
            area=1*u.m**2, binned=False)
        cts_a0V = Observation(sp_a0V, f_r).countrate(
            area=1*u.m**2, binned=False)
        cts_vega = Observation(sp_vega, f_r).countrate(
            area=1*u.m**2, binned=False)
        print("Flux from integrate K5V*%s %s" % (filtername, cts_k5V_i))
        print("Flux from integrate A0V*%s %s" % (filtername, cts_a0V_i))
        print("Flux from integrate VEGA*%s %s" % (filtername, cts_vega_i))
        print("Flux from Observation K5V*%s %s" % (filtername, cts_k5V))
        print("Flux from Observation A0V*%s %s" % (filtername, cts_a0V))
        print("Flux from Observation VEGA*%s %s" % (filtername, cts_vega))

        assert_quantity_allclose(cts_vega, cts_k5V, rtol=0.001)
        assert_quantity_allclose(cts_a0V, cts_k5V, rtol=0.001)

    def test_integrate_with_proper_waveset(self):
        '''
        Pay attention to integrate() with spectra
        '''
        # Assuming Observation computes correct values
        filtername = Filters.ESO_ETC_R
        f_r = Filters.get(filtername)
        sp_k5V = get_normalized_star_spectrum('K5V', 0, filtername)
        cts_observation = Observation(sp_k5V, f_r).countrate(
            area=1*u.m**2, binned=False)
        cts_integrate_filter_waveset = (sp_k5V*f_r).integrate(f_r.waveset)
        cts_integrate_star_waveset = (sp_k5V*f_r).integrate(sp_k5V.waveset)
        cts_integrate_spectrum_waveset = (
            sp_k5V*f_r).integrate((sp_k5V*f_r).waveset)

        print(" ---------- Flux for K5V*%s --------" % filtername)
        print("Flux from Observation K5V*%s %s" %
              (filtername, cts_observation))
        print("Flux from integrate on filter waveset %s" %
              (cts_integrate_filter_waveset))
        print("Flux from integrate on star waveset %s" %
              (cts_integrate_star_waveset))
        print("Flux from integrate on spectrum waveset %s" %
              (cts_integrate_spectrum_waveset))




if __name__ == "__main__":
    unittest.main()