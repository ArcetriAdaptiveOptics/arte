import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, CoatingsTransmissiveElementsCatalog, \
    GlassesTransmissiveElementsCatalog, DetectorsTransmissiveElementsCatalog


def plot_aluminium_mirror_v002():
    alu = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    wv = alu.waveset
    plt.plot(wv.to(u.um), alu.reflectance(wv), label='Aluminium')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()


def plot_silver_mirror_v002():
    ag = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    wv = ag.waveset
    plt.plot(wv.to(u.um), ag.reflectance(wv), label='Protected silver')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()

    
def plot_schmidt_plate_v004():
    plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_broadband_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_internal_001()
    wv = plate.waveset
    plt.plot(wv.to(u.um), plate.transmittance(wv), label='Correcting plate')
    plt.plot(wv.to(u.um), ar_coating.transmittance(wv),
             label='Broadband AR coating')
    plt.plot(wv.to(u.um), substrate.transmittance(wv),
             label='Suprasil 3002 (85 mm)')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()
    
    
def plot_lgs_dichroic_v005():
    lgs_dich = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    lma_coating = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3002_80mm_internal_001()
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_002()

    wv = lgs_dich.waveset
    plt.plot(wv.to(u.um), lgs_dich.transmittance(wv), label='LGS dichroic')
    plt.plot(wv.to(u.um), lma_coating.transmittance(wv),
             label='LMA coating')
    plt.plot(wv.to(u.um), substrate.transmittance(wv),
             label='Suprasil 3002 (80 mm)')
    plt.plot(wv.to(u.um), ar_coating.transmittance(wv),
             label='AR coating')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()


def plot_lma_coating_v001():
    lma_coating = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    wv = lma_coating.waveset
    plt.plot(wv.to(u.um), lma_coating.transmittance(wv),
        label='Transmittance')
    plt.plot(wv.to(u.um), lma_coating.reflectance(wv),
        label='Reflectance')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend(loc='lower right')


def plot_lgso_fm_v002():
    lgso_fm = MorfeoTransmissiveElementsCatalog.lgso_fm_002()
    wv = lgso_fm.waveset
    plt.plot(wv.to(u.um), lgso_fm.reflectance(wv))
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()


def plot_lgso_l1_v001():
    lgso_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    wv = lgso_l1.waveset
    ar_coat = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
    supra = GlassesTransmissiveElementsCatalog.suprasil3002_108mm_internal_001()
    plt.plot(wv.to(u.um), supra.transmittance(wv),
        label='Suprasil 3002 (108 mm)')
    plt.plot(wv.to(u.um), ar_coat.transmittance(wv),
        label='Narrowband (589 nm) AR coating')
    plt.plot(wv.to(u.um), lgso_l1.transmittance(wv), linewidth=0.8,
        label='LGSO-L1')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()
    

def plot_lgs_wfs_v001():
    lgs_wfs = MorfeoTransmissiveElementsCatalog.lgs_wfs_001()
    wv = lgs_wfs.waveset
    plt.plot(wv.to(u.um), lgs_wfs.transmittance(wv))
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.4, 1)
    plt.grid()

    
def plot_visir_dichroic_v002():
    visir_dich = MorfeoTransmissiveElementsCatalog.visir_dichroic_002()
    wv = visir_dich.waveset
    lzh_coating = CoatingsTransmissiveElementsCatalog.lzh_coating_for_visir_dichroic_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_amus_001()
    
    plt.plot(wv.to(u.um), visir_dich.transmittance(wv),
        label='VISIR dichroic')
    plt.plot(wv.to(u.um), lzh_coating.transmittance(wv),
        label='LZH coating', linewidth=0.9)
    plt.plot(wv.to(u.um), substrate.transmittance(wv),
             label='Fused silica (3 mm)', linewidth=0.9)
    plt.plot(wv.to(u.um), ar_coating.transmittance(wv),
             label='AR coating', linewidth=0.9)
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()


def plot_lo_collimator_v001():
    lo_coll = MorfeoTransmissiveElementsCatalog.lowfs_collimator_doublet_001()
    wv = lo_coll.waveset
    swir_coating = CoatingsTransmissiveElementsCatalog.ar_coating_swir_001()
    substrate1 = GlassesTransmissiveElementsCatalog.ohara_SFPL51_10mm_internal_001()
    substrate2 = GlassesTransmissiveElementsCatalog.ohara_PBL35Y_3mm_internal_001()
    
    plt.plot(wv.to(u.um), lo_coll.transmittance(wv),
        label='LO collimator')
    plt.plot(wv.to(u.um), swir_coating.transmittance(wv),
        label='SWIR AR coating')
    plt.plot(wv.to(u.um), substrate1.transmittance(wv),
             label='Ohara SFPL-51 (10 mm)')
    plt.plot(wv.to(u.um), substrate2.transmittance(wv),
             label='Ohara PBL-35Y (3 mm)')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.ylim(0.9, 1.005)
    plt.grid()
    plt.legend()


def plot_rwfs_collimator_v002():
    r_coll = MorfeoTransmissiveElementsCatalog.refwfs_collimator_doublet_002()
    wv = r_coll.waveset
    nir_i_coating = CoatingsTransmissiveElementsCatalog.ar_coating_nir_i_001()
    substrate1 = GlassesTransmissiveElementsCatalog.ohara_SFTM16_3mm_internal_001()
    substrate2 = GlassesTransmissiveElementsCatalog.cdgm_HQK3L_7mm_internal_001()
    
    plt.plot(wv.to(u.um), r_coll.transmittance(wv),
        label='R WFS collimator')
    plt.plot(wv.to(u.um), nir_i_coating.transmittance(wv),
        label='NIR I AR coating')
    plt.plot(wv.to(u.um), substrate1.transmittance(wv),
             label='Ohara S-FTM16 (3 mm)')
    plt.plot(wv.to(u.um), substrate2.transmittance(wv),
             label='CDGM H-QK3L (7 mm)')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.ylim(0.65, 1.01)
    plt.grid()
    plt.legend()
    
    
def plot_lo_adc_v002():
    lo_adc = MorfeoTransmissiveElementsCatalog.lowfs_adc_002()
    wv = lo_adc.waveset    
    swir_coating = CoatingsTransmissiveElementsCatalog.ar_coating_swir_001()
    substrate1 = GlassesTransmissiveElementsCatalog.schott_NSF2_9dot8_mm_internal_001()
    substrate2 = GlassesTransmissiveElementsCatalog.schott_NPSK53A_10_mm_internal_001()

    plt.plot(wv.to(u.um), lo_adc.transmittance(wv),
        label='LO ADC')
    plt.plot(wv.to(u.um), swir_coating.transmittance(wv),
        label='SWIR AR coating')
    plt.plot(wv.to(u.um), substrate1.transmittance(wv),
             label='Schott N-SF2 (9.8 mm)')
    plt.plot(wv.to(u.um), substrate2.transmittance(wv),
             label='Schott N-PSK53A (10 mm)')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.ylim(0.7, 1.005)
    plt.grid()
    plt.legend()


def plot_lo_lenslet_array_v001():
    lo_lenslet = MorfeoTransmissiveElementsCatalog.lowfs_lenslet_001()
    wv = lo_lenslet.waveset
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_amus_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
    
    plt.plot(wv.to(u.um), lo_lenslet.transmittance(wv),
        label='LO lenslet array')
    plt.plot(wv.to(u.um), ar_coating.transmittance(wv),
        label='AR coating')
    plt.plot(wv.to(u.um), substrate.transmittance(wv),
             label='Suprasil 3001 (3 mm)')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.ylim(0.93, 1.005)
    plt.grid()
    plt.legend()

    
def plot_rwfs_lenslet_array_v001():
    r_lenslet = MorfeoTransmissiveElementsCatalog.rwfs_lenslet_001()
    wv = r_lenslet.waveset
    nir_i_ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_nir_i_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3001_3mm_internal_001()
    
    plt.plot(wv.to(u.um), r_lenslet.transmittance(wv),
        label='R WFS lenslet array')
    plt.plot(wv.to(u.um), nir_i_ar_coating.transmittance(wv),
        label='NIR I AR coating')
    plt.plot(wv.to(u.um), substrate.transmittance(wv),
             label='Fused silica (2.15 mm)')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.ylim(0.6, 1.01)
    plt.grid()
    plt.legend()

    
def plot_cred1_cold_filters_v001():
    filt = DetectorsTransmissiveElementsCatalog.c_red_one_filters_001()
    wv = filt.waveset
    filtH = DetectorsTransmissiveElementsCatalog.c_red_one_H_filter_001()
    filtK = DetectorsTransmissiveElementsCatalog.c_red_one_K_filter_001()

    plt.plot(wv.to(u.um), filt.transmittance(wv),
        label='2H + 2K')
    plt.plot(wv.to(u.um), filtH.transmittance(wv),
        label='H')
    plt.plot(wv.to(u.um), filtK.transmittance(wv),
             label='K')
    plt.ylabel('Throughput')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.5, 2.5)
    plt.grid()
    plt.legend()

    
def plot_cred1_qe():
    c_red1 = DetectorsTransmissiveElementsCatalog.c_red_one_qe_001()
    wv = c_red1.waveset
    plt.plot(wv.to(u.um), c_red1.transmittance(wv))
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('QE')
    plt.grid()
    

def plot_ccd220_qe():
    ccd220 = DetectorsTransmissiveElementsCatalog.ccd220_qe_003()
    wv = ccd220.waveset
    plt.plot(wv.to(u.um), ccd220.transmittance(wv))
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('QE')
    plt.grid()
