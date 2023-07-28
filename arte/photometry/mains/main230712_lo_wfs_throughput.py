import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.morfeo_transmissive_systems import MorfeoLowOrderChannelTransmissiveSystem_002, \
    MorfeoLowOrderChannelTransmissiveSystem_001, \
    MorfeoLowOrderChannelTransmissiveSystem_003
from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, CoatingsTransmissiveElementsCatalog, \
    GlassesTransmissiveElementsCatalog, DetectorsTransmissiveElementsCatalog
from arte.photometry.eso_sky_calc import EsoSkyCalc
from arte.photometry import morfeo_transmissive_systems

WV_MIN_H = 1.490 * u.um
WV_MAX_H = 1.780 * u.um


def LO_WFS_throughput_001():
    '''
    Presented in PDR.
    '''
    lo_wfs_te = MorfeoLowOrderChannelTransmissiveSystem_001().as_transmissive_element()
    wv = lo_wfs_te.waveset
    t = lo_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.8, 2.)
    id_min = np.argwhere(wv == WV_MIN_H.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_H.to(u.Angstrom))[0][0]
    t_H = t[id_min:id_max]
    print('\nH band average transmission: %s' % np.mean(t_H))


def LO_WFS_throughput_002():
    '''
    Design for FDR.
    '''
    lo_wfs_te = MorfeoLowOrderChannelTransmissiveSystem_002().as_transmissive_element()
    wv = lo_wfs_te.waveset
    t = lo_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.8, 2.)
    id_min = np.argwhere(wv == WV_MIN_H.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_H.to(u.Angstrom))[0][0]
    t_H = t[id_min:id_max]
    print('\nH band average transmission: %s' % np.mean(t_H))
    

def LO_WFS_throughput_003():
    '''
    Design for FDR with coating from LMA for the LGS dichroic.
    '''
    lo_wfs_te = MorfeoLowOrderChannelTransmissiveSystem_003().as_transmissive_element()
    wv = lo_wfs_te.waveset
    t = lo_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.8, 2.)
    id_min = np.argwhere(wv == WV_MIN_H.to(u.Angstrom))[0][0]
    id_max = np.argwhere(wv == WV_MAX_H.to(u.Angstrom))[0][0]
    t_H = t[id_min:id_max + 1]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission: %s' % np.mean(t_H))


def aluminium_mirror_throughput():
    al_mirror = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    wv = al_mirror.waveset
    r = al_mirror.reflectance(wv)
    id_min = np.where(np.isclose(np.array(wv), WV_MIN_H.to(u.Angstrom).value,
                                 atol=100))[0][0]
    id_max = np.where(np.isclose(np.array(wv), WV_MAX_H.to(u.Angstrom).value,
                                 atol=100))[0][0]
    r_H = r[id_min:id_max + 1]   
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission: %s' % np.mean(r_H))
    

def silver_mirror_throughput():
    ag_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    wv = ag_mirror.waveset
    r = ag_mirror.reflectance(wv)
    id_min = np.where(np.isclose(np.array(wv), WV_MIN_H.to(u.Angstrom).value,
                                 atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv), WV_MAX_H.to(u.Angstrom).value,
                                 atol=10))[0][0]
    r_H = r[id_min:id_max + 1]   
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission: %s' % np.mean(r_H))
    

def correcting_plate_throughput():
    c_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    ar_coating_broad = CoatingsTransmissiveElementsCatalog.ar_coating_broadband_001()
    supra3002 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_internal_001()
    
    wv = c_plate.waveset
    id_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission of correcting plate: %s'
          % np.mean(c_plate.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of AR coating: %s' % 
          np.mean(ar_coating_broad.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of Suprasil 3002: %s'
          % np.mean(supra3002.transmittance(wv)[id_min:id_max + 1]))
    
    
def lgs_dichroic_throughput():
    lgs_dich = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    wv = lgs_dich.waveset
    r = lgs_dich.reflectance(wv)
    id_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_min:id_max + 2])
    print('\nH band average reflectance of LGS dichroic: %s'
          % np.mean(r[id_min:id_max + 2]))
    
    plt.figure()
    plt.plot(wv.to(u.um), r)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    # plt.xlim(0.8, 2.)


def visir_dichroic_throughput():
    lzh_coating = CoatingsTransmissiveElementsCatalog.lzh_coating_for_visir_dichroic_001()
    wv = lzh_coating.waveset
    
    id_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission of VISIR LZH coating: %s'
          % np.mean(lzh_coating.reflectance(wv)[id_min:id_max + 1]))
    
    plt.figure()
    plt.plot(wv.to(u.um), lzh_coating.reflectance(wv))
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    # plt.xlim(0.8, 2.)
    
    
def collimator_throughput():
    coll = MorfeoTransmissiveElementsCatalog.lowfs_collimator_doublet_001()
    swir_coating = CoatingsTransmissiveElementsCatalog.ar_coating_swir_001()
    substrate1 = GlassesTransmissiveElementsCatalog.ohara_SFPL51_10mm_internal_001()
    substrate2 = GlassesTransmissiveElementsCatalog.ohara_PBL35Y_3mm_internal_001()    
    wv = coll.waveset
    id_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_H.to(u.Angstrom).value,
                                    atol=200))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission of LO WFS collimator: %s'
          % np.mean(coll.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of SWIR AR coating: %s' % 
          np.mean(swir_coating.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of SFPL51 substrate(10 mm): %s'
          % np.mean(substrate1.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of PBL35Y substrate (3 mm): %s'
          % np.mean(substrate2.transmittance(wv)[id_min:id_max + 1]))
    

def adc_throughput():
    adc = MorfeoTransmissiveElementsCatalog.lowfs_adc_002()
    swir_coating = CoatingsTransmissiveElementsCatalog.ar_coating_swir_001()
    substrate1 = GlassesTransmissiveElementsCatalog.schott_NSF2_9dot8_mm_internal_001()
    substrate2 = GlassesTransmissiveElementsCatalog.schott_NPSK53A_10_mm_internal_001()  
    wv = adc.waveset
    id_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_H.to(u.Angstrom).value,
                                    atol=800))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission of LO WFS ADC: %s'
          % np.mean(adc.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of SWIR AR coating: %s' % 
          np.mean(swir_coating.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of NSF2 substrate (9.8 mm): %s'
          % np.mean(substrate1.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of NPSK53A substrate (10 mm): %s'
          % np.mean(substrate2.transmittance(wv)[id_min:id_max + 1]))
    
    
def lenslet_array_throughput():
    lenslet = MorfeoTransmissiveElementsCatalog.lowfs_lenslet_001()
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_amus_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()  
    wv = lenslet.waveset
    id_min = np.where(np.isclose(np.array(wv),
                                    WV_MIN_H.to(u.Angstrom).value,
                                    atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                    WV_MAX_H.to(u.Angstrom).value,
                                    atol=1000))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission of LO WFS lenslet array: %s'
          % np.mean(lenslet.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of Aµs AR coating: %s'
          % np.mean(ar_coating.transmittance(wv)[id_min:id_max + 1]))
    print('\nH band average transmission of Suprasil 3001 (3 mm): %s'
          % np.mean(substrate.transmittance(wv)[id_min:id_max + 1]))


def cold_filters_throughput():
    c_red1_filters = DetectorsTransmissiveElementsCatalog.c_red_one_filters_001()
    wv = c_red1_filters.waveset
    id_min = np.where(np.isclose(np.array(wv),
                                WV_MIN_H.to(u.Angstrom).value,
                                atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                WV_MAX_H.to(u.Angstrom).value,
                                atol=10))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average transmission of cold filters: %s'
          % np.mean(c_red1_filters.transmittance(wv)[id_min:id_max + 1]))
    
    plt.figure()
    plt.plot(wv.to(u.um), c_red1_filters.transmittance(wv))
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    # plt.xlim(0.8, 2.)

    
def c_red1_qe():
    c_red1 = DetectorsTransmissiveElementsCatalog.c_red_one_qe_001()
    wv = c_red1.waveset
    id_min = np.where(np.isclose(np.array(wv),
                                WV_MIN_H.to(u.Angstrom).value,
                                atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(wv),
                                WV_MAX_H.to(u.Angstrom).value,
                                atol=10))[0][0]
    print(wv[id_min:id_max + 1])
    print('\nH band average QE of C-RED1: %s'
          % np.mean(c_red1.transmittance(wv)[id_min:id_max + 1]))
    
    plt.figure()
    plt.plot(wv.to(u.um), c_red1.transmittance(wv))
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    # plt.xlim(0.8, 2.)


def plot_throughput():
    zenith_angle = 30 * u.deg
    airmass = 1 / np.cos(zenith_angle.to(u.rad))
    sky_no_moon = EsoSkyCalc(airmass=airmass, incl_moon='N')
    
    import warnings
    warnings.filterwarnings('ignore')
    
    lowfs = morfeo_transmissive_systems.MorfeoLowOrderChannelTransmissiveSystem_003()

    def plot_between(x, y, label, alpha):
        plt.plot(x, y, label=label)
        plt.fill_between(x.value, y, alpha=alpha)
    
    wv = sky_no_moon.lam.to(u.um)
    plot_between(wv, sky_no_moon.trans, label='Sky (no moon)', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * lowfs.transmittance_from_to(0, 5)(wv), label='Sky/ELT', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * lowfs.transmittance_from_to(0, 12)(wv),
                 label='Sky/ELT/MORFEO up to LGS dichroic', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * lowfs.transmittance_from_to(0, 20)(wv),
                 label='Sky/ELT/LOR up to VISIR dichroic', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * lowfs.transmittance_from_to(0, 23)(wv),
                 label='Sky/ELT/LO-WFS up to LA', alpha=0.4)
    plot_between(wv, sky_no_moon.trans * lowfs.transmittance_from_to(0, 26)(wv),
                 label='Sky/ELT/LO-WFS', alpha=0.4)
    plt.legend(loc='upper left', fontsize='x-small')
    plt.xlabel('Wavelength [μm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.3, 2.5)
