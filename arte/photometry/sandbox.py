'''
Created on Oct 13, 2019

@author: lbusoni
'''
import matplotlib.pyplot as plt
import synphot
from astropy import units as u
from synphot import SourceSpectrum
from synphot.models import Box1D, Empirical1D
from synphot.spectrum import SpectralElement
from synphot.observation import Observation
from arte.photometry.filters import Filters
from synphot.units import FLAM
from arte.photometry import get_normalized_star_spectrum
from arte.photometry.eso_sky_calc import EsoSkyCalc
import numpy as np


def misc():
    # get_vega() downloads this one
    synphot.specio.read_remote_spec(
        'http://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_008.fits')

    # G5V of UVKLIB subset of Pickles library
    # see http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas.html
    synphot.specio.read_remote_spec(
        'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_27.fits')

    # read the sourcespectrum
    spG5V = SourceSpectrum.from_file(
        'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_27.fits')
    spG2V = SourceSpectrum.from_file(
        'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/pickles_uk_26.fits')

    filtR = SpectralElement.from_filter('johnson_r')
    spG2V_19r = spG2V.normalize(
        19 * synphot.units.VEGAMAG, filtR,
        vegaspec=SourceSpectrum.from_vega())

    bp = SpectralElement(Box1D, x_0=700 * u.nm, width=600 * u.nm)
    obs = Observation(spG2V_19r, bp)
    obs.countrate(area=50 * u.m ** 2)

    bp220 = SpectralElement(Box1D, x_0=800 * u.nm, width=400 * u.nm)
    bpCRed = SpectralElement(Box1D, x_0=1650 * u.nm, width=300 * u.nm) * 0.8

    Observation(spG2V_19r, bp220).countrate(area=50 * u.m ** 2)
    Observation(spG2V_19r, bpCRed).countrate(area=50 * u.m ** 2)

    spM0V_8R = get_normalized_star_spectrum('M0V', 8.0, 'johnson_r')

    uPhotonSecM2Micron = u.photon / (u.s * u.m ** 2 * u.micron)

    spG2V_8R = get_normalized_star_spectrum("G2V", 8.0, 'johnson_r')
    plt.plot(spG2V_8R.waveset, spG2V_8R(spG2V_8R.waveset).to(uPhotonSecM2Micron))

    # compare with Armando's
    spG2V_19R = get_normalized_star_spectrum("G2V", 19, 'johnson_r')
    bp = SpectralElement(Box1D, x_0=700 * u.nm, width=600 * u.nm)
    obs = Observation(spG2V_19R, bp)
    obs.countrate(area=50 * u.m ** 2)

    # zeropoint in filtro r in erg/s/cm2/A
    Observation(get_normalized_star_spectrum('A0V', 0, 'johnson_r'), SpectralElement.from_filter('johnson_r')).effstim('flam')
    # zeropoint in ph/s/m2
    Observation(get_normalized_star_spectrum('A0V', 0, 'johnson_r'), SpectralElement.from_filter('johnson_r')).countrate(area=1 * u.m ** 2)


def check_zeropoints_ESO():
    '''
    Shouldn't they match http://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html#mags?
    '''
    # sky = EsoSkyCalc(airmass=1, moon_target_sep=45.0, moon_alt=45,
    #                 observatory='paranal', wdelta=10)
    # atmo = SpectralElement(Empirical1D, points=sky.lam, lookup_table=sky.trans)

    filters = [Filters.ESO_ETC_U,
               Filters.ESO_ETC_B,
               Filters.ESO_ETC_V,
               Filters.ESO_ETC_R,
               Filters.ESO_ETC_I,
               Filters.ESO_ETC_Z,
               Filters.ESO_ETC_Y,
               Filters.ESO_ETC_J,
               Filters.ESO_ETC_H,
               Filters.ESO_ETC_K,
               Filters.ESO_ETC_L,
               Filters.ESO_ETC_M
               ]

    eso_etc_zp = np.array([
        4.18023e-09, 6.60085e-09, 3.60994e-09,
        2.28665e-09, 1.22603e-09, 7.76068e-10,
        5.973e-10, 3.12e-10, 1.14e-10, 3.94e-11,
        4.83e-12, 2.04e-12, 1.23e-13, 6.8e-15]) * FLAM

    star = get_normalized_star_spectrum('vega', 0.0, Filters.ESO_ETC_R)
    for filt_name, etc_zp in zip(filters, eso_etc_zp):
        filt = Filters.get(filt_name)
        obs = Observation(star, filt)
        zp = obs.effstim('flam', wavelengths=filt.waveset)
        err = (zp - etc_zp) / etc_zp * 100
        cnts = obs.countrate(area=1 * u.m ** 2, wavelengths=filt.waveset)
        print(f"{filt_name:10s} | {zp:10.3e} | {err:+7.3f}% | {cnts:10.3e} |"
              f"{filt.waveset.to(u.nm)[0]:g} {filt.waveset.to(u.nm)[-1]:g}")






