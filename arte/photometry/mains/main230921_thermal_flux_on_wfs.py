import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.morfeo_transmissive_systems import MorfeoLowOrderChannelTransmissiveSystem_003, \
    MorfeoReferenceChannelTransmissiveSystem_004
from arte.photometry.transmissive_elements_catalogs import DetectorsCatalog, \
    MorfeoTransmissiveElementsCatalog
from morfeo.utils.constants import MORFEO
from synphot.spectrum import SourceSpectrum
from synphot.models import BlackBody1D
from synphot import units


def thermal_flux_on_LO_WFS():
    T = 278.15 * u.K
    cw_ccd_distance = 38.83 * u.mm
    cold_stop_oversizing = 0.65
    bb_spectrum = SourceSpectrum(BlackBody1D, temperature=T)
    solid_angle_pup = np.pi / 4 * MORFEO.lo_wfs_pupil_diameter ** 2 / (
        MORFEO.lo_wfs_lenslet_focal_length ** 2 + (
            MORFEO.lo_wfs_pupil_diameter / 2) ** 2) * u.sr
    solid_angle_cam = np.pi / 4 * (
        MORFEO.lo_wfs_pupil_diameter * (1 + cold_stop_oversizing)) ** 2 / (
        cw_ccd_distance ** 2 + (MORFEO.lo_wfs_pupil_diameter * (
            1 + cold_stop_oversizing) / 2) ** 2) * u.sr
                        
    lo_wfs_transmissive_system = MorfeoLowOrderChannelTransmissiveSystem_003()
    # remove CRED1 QE from path
    lo_wfs_transmissive_system.remove(-1)
    # remove cold filters from path
    lo_wfs_transmissive_system.remove(-1)
    # remove lenslet array from path
    lo_wfs_transmissive_system.remove(-1)
    waveset = lo_wfs_transmissive_system._waveset
    lenslet_array = MorfeoTransmissiveElementsCatalog.lowfs_lenslet_001()
    cold_filters = DetectorsCatalog.c_red_one_filters_001()
    cred1_qe = DetectorsCatalog.c_red_one_qe_001()    

    th_flux = bb_spectrum(waveset) * (
                lo_wfs_transmissive_system.emissivity(waveset) * solid_angle_pup + \
                1 * (solid_angle_cam - solid_angle_pup)) * \
            (lenslet_array.transmittance * cold_filters.transmittance * \
            cred1_qe.transmittance)(waveset) * (
                (MORFEO.lo_wfs_px_size).to(u.cm / u.pix)
                ) ** 2 * u.electron / u.s / u.cm ** 2 / u.Angstrom / units.PHOTLAM
        
    plt.semilogy(waveset.to(u.um), th_flux)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Thermal flux [e-/s/px/µm]')
    plt.grid()
    plt.xlim(0.6, 4.4)
    print(np.trapz(th_flux, waveset.to(u.Angstrom)))
    
    return th_flux


def thermal_flux_on_R_WFS():
    T = 278.15 * u.K
    bb_spectrum = SourceSpectrum(BlackBody1D, temperature=T)
    solid_angle = np.pi / 4 * MORFEO.r_wfs_pupil_diameter ** 2 / (
        MORFEO.r_wfs_lenslet_focal_length ** 2 + (
            MORFEO.r_wfs_pupil_diameter / 2) ** 2) * u.sr
    r_wfs_transmissive_system = MorfeoReferenceChannelTransmissiveSystem_004()
    # remove ALICE QE from path
    r_wfs_transmissive_system.remove(-1)
    # remove ALICE window from path
    r_wfs_transmissive_system.remove(-1)
    # remove lenslet array from path
    r_wfs_transmissive_system.remove(-1)
    waveset = r_wfs_transmissive_system._waveset
    lenslet_array = MorfeoTransmissiveElementsCatalog.rwfs_lenslet_001()
    alice_window = MorfeoTransmissiveElementsCatalog.alice_entrance_window_001()
    qe = DetectorsCatalog.ccd220_qe_003()    

    th_flux = (
            bb_spectrum * r_wfs_transmissive_system.emissivity * \
            lenslet_array.transmittance * alice_window.transmittance * \
            qe.transmittance)(waveset) * solid_angle / u.sr * (
                (MORFEO.r_wfs_px_size).to(u.cm / u.pix)
                ) ** 2 * u.electron / u.s / u.cm ** 2 / u.Angstrom / units.PHOTLAM
        
    plt.semilogy(waveset.to(u.um), th_flux)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Thermal flux [e-/s/px/µm]')
    plt.grid()
    plt.xlim(0.5, 1.5)
    
    print(np.trapz(th_flux, waveset.to(u.Angstrom)))
    return th_flux, waveset
