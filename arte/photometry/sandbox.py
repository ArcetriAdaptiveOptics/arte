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
from arte.photometry.transmissive_elements_catalogs import MorfeoTransmissiveElementsCatalog, \
    GlassesTransmissiveElementsCatalog
from scipy import interpolate


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


def external_transmittance_calculator(l1, l2, t1, a):
    '''
    Use external transmittance data of a glass with thickness l2 to compute
    external transmittance of a same glass but with different thickness l1.
    The computation is based on the equation for the external transmittance:
    
    T = (1 - R**2) * exp(-a * l)
    
    where R is the reflectance, a is the attenuation coefficient and l is the
    thickness of the glass. If we consider two different values of thickness, 
    thus of transmittance, we can compute the unknown transmittance T2 as:
    
    T2 = T1 *  exp(-a * (l2 - l1))

    '''
    t2 = t1 * np.exp(-a * (l2 - l1))
    return t2


def internal_transmittance_calculator(l1, l2, t1):
    '''
    Use internal transmittance data t1 of a glass with thickness l1 to compute 
    internal transmittance t2 of a same glass but with different thickness l2.
    The transmittance is computed with the following equation:
    
        t2 = t1**(l2/l1)
    '''
    t2 = t1 ** (l2 / l1)
    return t2


def attenuation_coefficient_calculator(l1, l2, t1, t2):
    '''
    Compute attenuation coefficient of a glass from external transmittance data
    of two different values of thickness.
    The computation is based on the equation for the external transmittance
    of a glass:
    
    T = (1 - R**2) * exp(-a * l)
    
    where R is the reflectance, a is the attenuation coefficient and l is the
    thickness of the glass. If we consider two different values of thickness, 
    thus of transmittance, we can compute the ratio between the transmittances:
    
    T1 / T2 = exp(-a * (l1 - l2))
    
    and from this equation we can derive the attenuation coefficient as:
    
    a = (lnT2 - lnT1) / (l1 - l2)
    '''
    a = (np.log(t2) - np.log(t1)) / (l1 - l2)
    return a
   
   
def main230414_derive_CPM_transmittance_from_Demetrio_plot():
    data = np.array(
        [[0.50, 0.98], [0.52, 0.985], [0.55, 0.99], [0.58, 0.985], [0.6, 0.984],
          [0.65, 0.982], [0.7, 0.985], [0.78, 0.984], [0.8, 0.985],
          [0.85, 0.985], [0.9, 0.984], [0.95, 0.982], [1., 0.9825],
          [1.1, 0.985], [1.2, 0.986], [1.3, 0.987], [1.4, 0.989], [1.5, 0.99],
          [1.6, 0.989], [1.7, 0.987], [1.8, 0.988], [1.9, 0.989], [2.0, 0.99],
          [2.1, 0.992], [2.2, 0.993], [2.3, 0.99], [2.4, 0.987]])
    f_interp = interpolate.interp1d(data[:, 0], data[:, 1], kind='cubic')
    new_wv = np.linspace(0.5, 2.4, 96)
    data_interp = f_interp(new_wv)
    plt.plot(new_wv, data_interp, label='Interp')
    plt.plot(data[:, 0], data[:, 1], '-.', label='From plot')
    plt.grid()
    plt.legend()
    # tosave = np.stack((new_wv, data_interp), axis=1)
    # folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/ar_broadband_001/'
    # np.savetxt(folder + 't.dat', tosave, fmt='%1.4f')
    return new_wv, data_interp


def main230202_compute_attenuation_coefficient_of_suprasil():
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_001()
    supra85 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_001()
    wv = supra10.waveset
    t1 = supra85.transmittance(wv)
    t2 = supra10.transmittance(wv)
    l1 = 85 * u.mm
    l2 = 10 * u.mm
    a_supra = attenuation_coefficient_calculator(l1, l2, t1, t2)
    plt.plot(wv.to(u.um), a_supra)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('a [mm$^{-1}]$')
    return a_supra, wv


def main230202_compute_transmittance_of_suprasil3002_80mm_and_save_dat():
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_001()
    supra85 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_001()
    wv = supra10.waveset
    t1 = supra85.transmittance(wv)
    t2 = supra10.transmittance(wv)
    l1 = 85 * u.mm
    l2 = 10 * u.mm
    a_supra = attenuation_coefficient_calculator(l1, l2, t1, t2)
    l3 = 80 * u.mm
    t3 = external_transmittance_calculator(l1, l3, t1, a_supra)
    plt.plot(wv.to(u.um), t3)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t3), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_80mm_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t3, wv


def main230202_compute_transmittance_of_suprasil3002_108mm_and_save_dat():
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_001()
    supra85 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_001()
    wv = supra10.waveset
    t1 = supra85.transmittance(wv)
    t2 = supra10.transmittance(wv)
    l1 = 85 * u.mm
    l2 = 10 * u.mm
    a_supra = attenuation_coefficient_calculator(l1, l2, t1, t2)
    l3 = 108 * u.mm
    t3 = external_transmittance_calculator(l1, l3, t1, a_supra)
    plt.plot(wv.to(u.um), t3)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t3), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_108mm_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t3, wv


def main230202_compute_transmittance_of_suprasil3002_40mm_and_save_dat():
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_001()
    supra85 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_001()
    wv = supra10.waveset
    t1 = supra85.transmittance(wv)
    t2 = supra10.transmittance(wv)
    l1 = 85 * u.mm
    l2 = 10 * u.mm
    a_supra = attenuation_coefficient_calculator(l1, l2, t1, t2)
    l3 = 40 * u.mm
    t3 = external_transmittance_calculator(l1, l3, t1, a_supra)
    plt.plot(wv.to(u.um), t3)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t3), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_40mm_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t3, wv


def main230202_compute_transmittance_of_suprasil3002_60mm_and_save_dat():
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_001()
    supra85 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_001()
    wv = supra10.waveset
    t1 = supra85.transmittance(wv)
    t2 = supra10.transmittance(wv)
    l1 = 85 * u.mm
    l2 = 10 * u.mm
    a_supra = attenuation_coefficient_calculator(l1, l2, t1, t2)
    l3 = 60 * u.mm
    t3 = external_transmittance_calculator(l1, l3, t1, a_supra)
    plt.plot(wv.to(u.um), t3)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t3), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_60mm_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t3, wv


def main230202_compute_transmittance_of_suprasil3002_70mm_and_save_dat():
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_001()
    supra85 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_001()
    wv = supra10.waveset
    t1 = supra85.transmittance(wv)
    t2 = supra10.transmittance(wv)
    l1 = 85 * u.mm
    l2 = 10 * u.mm
    a_supra = attenuation_coefficient_calculator(l1, l2, t1, t2)
    l3 = 70 * u.mm
    t3 = external_transmittance_calculator(l1, l3, t1, a_supra)
    plt.plot(wv.to(u.um), t3)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t3), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_70mm_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t3, wv


def main230203_compute_internal_transmittance_of_suprasil3002_80mm_and_save_dat(
        ):
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_internal_001()
    wv = supra10.waveset
    t1 = supra10.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 80 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_80mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t2, wv


def main230203_compute_internal_transmittance_of_suprasil3002_85mm_and_save_dat(
        ):
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_internal_001()
    wv = supra10.waveset
    t1 = supra10.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 85 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_85mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t2, wv


def main230203_compute_internal_transmittance_of_suprasil3002_108mm_and_save_dat(
        ):
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_internal_001()
    wv = supra10.waveset
    t1 = supra10.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 108 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_108mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t2, wv


def main230203_compute_internal_transmittance_of_suprasil3002_40mm_and_save_dat(
        ):
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_internal_001()
    wv = supra10.waveset
    t1 = supra10.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 40 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_40mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t2, wv


def main230203_compute_internal_transmittance_of_suprasil3002_60mm_and_save_dat(
        ):
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_internal_001()
    wv = supra10.waveset
    t1 = supra10.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 60 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_60mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t2, wv


def main230203_compute_internal_transmittance_of_suprasil3002_70mm_and_save_dat(
        ):
    supra10 = GlassesTransmissiveElementsCatalog.suprasil3002_10mm_internal_001()
    wv = supra10.waveset
    t1 = supra10.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 70 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/suprasil3002_70mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save)
    return t2, wv
