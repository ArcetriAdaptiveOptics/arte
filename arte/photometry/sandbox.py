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
from arte.photometry.transmittance_calculator import internal_transmittance_calculator, \
    attenuation_coefficient_calculator, external_transmittance_calculator


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


def main230420_derive_VISIR_dichroic_transmittance_from_LZH():
    data = np.array(
        [[0.590, 0.95], [0.594, 0.988], [0.600, 0.978], [0.605, 0.979],
         [0.610, 0.996], [0.614, 0.978], [0.618, 0.995], [0.621, 0.977],
         [0.625, 0.999], [0.630, 0.973], [0.635, 0.999], [0.640, 0.975],
         [0.645, 0.997], [0.650, 0.976], [0.655, 0.996], [0.660, 0.975],
         [0.665, 0.998], [0.670, 0.979], [0.675, 0.992], [0.680, 0.979],
         [0.685, 0.992], [0.690, 0.976], [0.695, 0.993], [0.705, 0.978],
         [0.710, 0.995], [0.720, 0.978], [0.725, 0.982], [0.730, 0.979],
         [0.735, 0.985], [0.740, 0.978], [0.750, 0.995], [0.757, 0.990],
         [0.763, 0.994], [0.770, 0.979], [0.778, 0.991], [0.786, 0.979],
         [0.795, 0.991], [0.800, 0.978], [0.810, 0.995], [0.820, 0.979],
         [0.825, 0.993], [0.835, 0.979], [0.843, 0.995], [0.855, 0.978],
         [0.863, 0.993], [0.872, 0.979], [0.881, 0.994], [0.892, 0.979],
         [0.901, 0.995], [0.912, 0.979], [0.922, 0.995], [0.934, 0.978],
         [0.944, 0.990], [0.955, 0.979], [0.965, 0.992], [0.975, 0.978],
         [0.987, 0.996], [1.00, 0.978], [1.01, 0.997], [1.02, 0.95]
         ])
    plt.plot(data[:, 0], data[:, 1], '.-')
    plt.grid()
    tosave = np.stack((data[:, 0], data[:, 1]), axis=1)
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/morfeo/visir_dichroic_002/'
    np.savetxt(folder + 't.dat', tosave, fmt='%1.4f')


def main230420_derive_SWIR_AR_coating_from_EdmundOptics_plot():
    data_perc = np.array(
        [[0.90, 0.40], [0.95, 0.25], [1.0, 0.35], [1.1, 0.50], [1.2, 0.49],
          [1.25, 0.45], [1.3, 0.40], [1.35, 0.38], [1.4, 0.35],
          [1.45, 0.40], [1.5, 0.44], [1.55, 0.5], [1.60, 0.7],
          [1.65, 0.75], [1.7, 1]])
    f_interp = interpolate.interp1d(data_perc[:, 0], data_perc[:, 1] / 100, kind='cubic')
    new_wv = np.linspace(0.9, 1.7, 81)
    data_interp = f_interp(new_wv)
    plt.plot(new_wv, data_interp, label='Interp')
    plt.plot(data_perc[:, 0], data_perc[:, 1] / 100, '-.', label='From plot')
    plt.grid()
    plt.legend()
    tosave = np.stack((new_wv, data_interp), axis=1)
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/ar_swir_001/'
    np.savetxt(folder + 'r.dat', tosave, fmt='%1.4f')
    return new_wv, data_interp


def main230419_compute_internal_transmittance_of_ohara_PBL35Y_3mm_and_save_dat(
        ):
    pbl_10mm = GlassesTransmissiveElementsCatalog.ohara_PBL35Y_10mm_internal_001()
    wv = pbl_10mm.waveset
    t1 = pbl_10mm.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 3 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/ohara_PBL35Y_3mm_internal_001/'
    np.savetxt(folder + 't.dat', to_save, fmt='%1.4f')
    return t1, t2, wv 


def main230414_compute_internal_transmittance_of_ohara_quartz_SK1300_85mm_and_save_dat(
        ):
    sk1300_10mm = GlassesTransmissiveElementsCatalog.ohara_quartz_SK1300_10mm_internal_001()
    wv = sk1300_10mm.waveset
    t1 = sk1300_10mm.transmittance(wv)
    l1 = 10 * u.mm
    l2 = 85 * u.mm
    t2 = internal_transmittance_calculator(l1, l2, t1)
    plt.plot(wv.to(u.um), t2)
    plt.grid()
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transmittance')
    to_save = np.stack((wv.to(u.um).value, t2), axis=1).value
    folder = '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/glasses/ohara_quartz_SK1300_85mm_001/'
    np.savetxt(folder + 't.dat', to_save, fmt='%1.4f')
    return t1, t2, wv 

   
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
