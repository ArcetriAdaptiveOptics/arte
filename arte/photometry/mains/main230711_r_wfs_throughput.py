import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.morfeo_transmissive_systems import MorfeoReferenceChannelTransmissiveSystem_001, \
    MorfeoReferenceChannelTransmissiveSystem_002, \
    MorfeoReferenceChannelTransmissiveSystem_003
from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, CoatingsTransmissiveElementsCatalog, \
    GlassesTransmissiveElementsCatalog, DetectorsTransmissiveElementsCatalog
from arte.photometry import morfeo_transmissive_systems
from arte.photometry.eso_sky_calc import EsoSkyCalc

WV_MIN_RI = 0.6 * u.um
WV_MAX_RI = 1.0 * u.um


def R_WFS_001_throughput():
    '''
    Presented in PDR.
    '''
    r_wfs_te = MorfeoReferenceChannelTransmissiveSystem_001().as_transmissive_element()
    wv = r_wfs_te.waveset
    t = r_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    id_min = np.argwhere(wv == WV_MIN_RI.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_RI.to(u.Angstrom))[0][0]
    t_RI = t[id_min:id_max]
    print('\nR+I band average transmission: %s' % np.mean(t_RI))


def R_WFS_002_throughput():
    '''
    Design for FDR. CCD220 QE is related to deep depleted silicon case.
    '''
    r_wfs_te = MorfeoReferenceChannelTransmissiveSystem_002().as_transmissive_element()
    wv = r_wfs_te.waveset
    t = r_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    id_min = np.argwhere(wv == WV_MIN_RI.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_RI.to(u.Angstrom))[0][0]
    t_RI = t[id_min:id_max]
    print('\nR+I band average transmission: %s' % np.mean(t_RI))

    
def R_WFS_003_throughput():
    '''
    Design for FDR. CCD220 QE is conservative (minimum ESO requirement).
    '''
    r_wfs_te = MorfeoReferenceChannelTransmissiveSystem_003().as_transmissive_element()
    wv = r_wfs_te.waveset
    t = r_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    id_min = np.argwhere(wv == WV_MIN_RI.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_RI.to(u.Angstrom))[0][0]
    t_RI = t[id_min:id_max]
    print('\nR+I band average transmission: %s' % np.mean(t_RI))

    
def aluminium_mirror_throughput():
    al_mirror = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    wv = al_mirror.waveset
    r = al_mirror.reflectance(wv)
    id_min = np.where(np.isclose(np.array(wv), WV_MIN_RI.to(u.Angstrom).value,
                                 atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv), WV_MAX_RI.to(u.Angstrom).value,
                                 atol=20))[0][0]
    r_RI = r[id_min:id_max + 1]   
    print(wv[id_min:id_max + 1])
    print('\nR+I band average transmission: %s' % np.mean(r_RI))
    

def silver_mirror_throughput():
    ag_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    wv = ag_mirror.waveset
    r = ag_mirror.reflectance(wv)
    id_min = np.where(np.isclose(np.array(wv), WV_MIN_RI.to(u.Angstrom).value,
                                 atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv), WV_MAX_RI.to(u.Angstrom).value,
                                 atol=10))[0][0]
    r_RI = r[id_min:id_max + 1]   
    print(wv[id_min:id_max + 1])
    print('\nR+I band average transmission: %s' % np.mean(r_RI))
    

def correcting_plate_throughput():
    c_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    ar_coating_broad = CoatingsTransmissiveElementsCatalog.ar_coating_broadband_001()
    supra3002 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_internal_001()
    
    wv = c_plate.waveset
    id_RI_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_RI_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_RI_min:id_RI_max + 1])
    print('\nR+I band average transmission of correcting plate: %s'
          % np.mean(c_plate.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of AR coating: %s' % 
          np.mean(ar_coating_broad.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of Suprasil 3002: %s'
          % np.mean(supra3002.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    
    
def lgs_dichroic_throughput():
    lgs_dich = CoatingsTransmissiveElementsCatalog.materion_average_002()
    wv = lgs_dich.waveset
    r = lgs_dich.reflectance(wv)
    id_RI_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_RI_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_RI_min:id_RI_max + 2])
    print('\nR+I band average reflectance of LGS dichroic: %s'
          % np.mean(r[id_RI_min:id_RI_max + 2]))


def visir_dichroic_throughput():
    visir_dich = MorfeoTransmissiveElementsCatalog.visir_dichroic_002()
    wv = visir_dich.waveset
    lzh_coating = CoatingsTransmissiveElementsCatalog.lzh_coating_for_visir_dichroic_001()
    fused_silica_3mm = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_amus_001()
    
    id_RI_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_RI_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_RI_min:id_RI_max + 1])
    print('\nR+I band average transmission of VISIR dichroic: %s'
          % np.mean(visir_dich.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of VISIR LZH coating: %s'
          % np.mean(lzh_coating.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of substrate: %s' % 
          np.mean(fused_silica_3mm.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of VISIR AR coating: %s'
          % np.mean(ar_coating.transmittance(wv)[id_RI_min:id_RI_max + 1]))

  
def collimator_throughput():
    r_collimator = MorfeoTransmissiveElementsCatalog.refwfs_collimator_doublet_002()
    nir_ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_nir_i_001()
    sftm_3mm = GlassesTransmissiveElementsCatalog.ohara_SFTM16_3mm_internal_001()
    cdgm_7mm = GlassesTransmissiveElementsCatalog.cdgm_HQK3L_7mm_internal_001()
    wv = r_collimator.waveset
    
    id_RI_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_RI_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_RI_min:id_RI_max + 1])
    print('\nR+I band average transmission of R WFS collimator: %s'
          % np.mean(r_collimator.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of NIR I AR coating: %s'
          % np.mean(nir_ar_coating.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of SFTM substrate: %s'
          % np.mean(sftm_3mm.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of CDGM substrate: %s'
          % np.mean(cdgm_7mm.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    
    
def ALICE_entrance_window_throughput():
    alice_ew = MorfeoTransmissiveElementsCatalog.alice_entrance_window_001()
    wv = alice_ew.waveset
    id_RI_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_RI_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_RI_min:id_RI_max + 1])
    print('\nR+I band average transmission of ALICE EW: %s'
          % np.mean(alice_ew.transmittance(wv)[id_RI_min:id_RI_max + 1]))

    
def lenslet_array_throughput():
    r_la = MorfeoTransmissiveElementsCatalog.rwfs_lenslet_001()
    nir_ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_nir_i_001()
    supra3001_3mm = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
    
    wv = r_la.waveset
    id_RI_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_RI_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_RI.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_RI_min:id_RI_max + 1])
    print('\nR+I band average transmission of R WFS lenslet array: %s'
          % np.mean(r_la.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of NIR I coating: %s'
          % np.mean(nir_ar_coating.transmittance(wv)[id_RI_min:id_RI_max + 1]))
    print('\nR+I band average transmission of fused silica: %s'
          % np.mean(supra3001_3mm.transmittance(wv)[id_RI_min:id_RI_max + 1]))

    
def ccd220_QE_001():
    ccd220_qe = DetectorsTransmissiveElementsCatalog.ccd220_qe_001()   
    wv = ccd220_qe.waveset
    id_min = np.argwhere(wv == WV_MIN_RI.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_RI.to(u.Angstrom))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nR+I band average QE of CCD220: %s'
          % np.mean(ccd220_qe.transmittance(wv)[id_min:id_max + 1]))
    
    
def ccd220_QE_002():
    ccd220_qe = DetectorsTransmissiveElementsCatalog.ccd220_qe_002()   
    wv = ccd220_qe.waveset
    id_min = np.argwhere(wv == WV_MIN_RI.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_RI.to(u.Angstrom))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nR+I band average QE of CCD220: %s'
          % np.mean(ccd220_qe.transmittance(wv)[id_min:id_max + 1]))
    
    
def ccd220_QE_003():
    ccd220_qe = DetectorsTransmissiveElementsCatalog.ccd220_qe_003()   
    wv = ccd220_qe.waveset
    id_min = np.argwhere(wv == WV_MIN_RI.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_RI.to(u.Angstrom))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nR+I band average QE of CCD220: %s'
          % np.mean(ccd220_qe.transmittance(wv)[id_min:id_max + 1]))
    
    
def plot_throughput():
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    sky_no_moon = EsoSkyCalc(airmass=airmass, incl_moon='N')
    
    import warnings
    warnings.filterwarnings('ignore')
    
    rwfs = morfeo_transmissive_systems.MorfeoReferenceChannelTransmissiveSystem_002()

    def plot_between(x, y, label, alpha):
        plt.plot(x, y, label=label)
        plt.fill_between(x.value, y, alpha=alpha)
    
    wv = sky_no_moon.lam.to(u.um)
    plot_between(wv, sky_no_moon.trans, label='Sky (no moon)', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * rwfs.transmittance_from_to(0, 5)(wv), label='Sky/ELT', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * rwfs.transmittance_from_to(0, 12)(wv),
                 label='Sky/ELT/MORFEO up to LGS dichroic', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * rwfs.transmittance_from_to(0, 20)(wv),
                 label='Sky/ELT/LOR up to VISIR dichroic', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * rwfs.transmittance_from_to(0, 25)(wv),
                 label='Sky/ELT/R-WFS up to LA', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * rwfs.transmittance_from_to(0, 26)(wv),
                 label='Sky/ELT/R-WFS', alpha=0.4)
    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('Wavelength [μm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.3, 2.5)
