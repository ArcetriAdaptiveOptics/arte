import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.morfeo_transmissive_systems import MorfeoLgsChannelTransmissiveSystem_003, \
    MorfeoLgsChannelTransmissiveSystem_005, \
    MorfeoLgsChannelTransmissiveSystem_004, \
    MorfeoLgsChannelTransmissiveSystem_006
from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, CoatingsTransmissiveElementsCatalog, \
    GlassesTransmissiveElementsCatalog, DetectorsTransmissiveElementsCatalog

WV_589NM = 0.589 * u.um


def LGS_WFS_003_throughput():
    lgs_wfs_te = MorfeoLgsChannelTransmissiveSystem_003().as_transmissive_element()
    wv = lgs_wfs_te.waveset
    t = lgs_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    print('\nThroughput at 589 nm: %s' % t.max())

    
def LGS_WFS_004_throughput():
    lgs_wfs_te = MorfeoLgsChannelTransmissiveSystem_004().as_transmissive_element()
    wv = lgs_wfs_te.waveset
    t = lgs_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    print('\nThroughput at 589 nm: %s' % t.max())


def LGS_WFS_005_throughput():
    lgs_wfs_te = MorfeoLgsChannelTransmissiveSystem_005().as_transmissive_element()
    wv = lgs_wfs_te.waveset
    t = lgs_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    print('\nThroughput at 589 nm: %s' % t.max())
    

def LGS_WFS_006_throughput():
    lgs_wfs_te = MorfeoLgsChannelTransmissiveSystem_006().as_transmissive_element()
    wv = lgs_wfs_te.waveset
    t = lgs_wfs_te.transmittance(wv)
    plt.plot(wv.to(u.um), t)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Throughput')
    plt.grid()
    plt.xlim(0.2, 1.5)
    print('\nThroughput at 589 nm: %s' % t.max())


def aluminium_mirror_throughput():
    al_mirror = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    wv = al_mirror.waveset
    r = al_mirror.reflectance(wv)
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                 atol=10))[0][0]
    print(wv[id_589nm])
    print('\nThroughput at 589 nm: %s' % r[id_589nm])


def silver_mirror_throughput():
    ag_mirror = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    wv = ag_mirror.waveset
    r = ag_mirror.reflectance(wv)
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                 atol=10))[0][0]
    print(wv[id_589nm])
    print('\nThroughput at 589 nm: %s' % r[id_589nm])


def correcting_plate_throughput():
    c_plate = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    ar_coating_broad = CoatingsTransmissiveElementsCatalog.ar_coating_broadband_001()
    supra3002 = GlassesTransmissiveElementsCatalog.suprasil3002_85mm_internal_001()
    
    wv = c_plate.waveset
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
    print(wv[id_589nm])
    print('\nCorrecting plate throughput at 589 nm: %s'
          % c_plate.transmittance(wv)[id_589nm])
    print('\nAR coating throughput at 589 nm: %s' % 
          ar_coating_broad.transmittance(wv)[id_589nm])
    print('\nSuprasil 3002 (85 mm) throughput at 589 nm: %s'
          % supra3002.transmittance(wv)[id_589nm])


def lgs_dichroic_throughput():
    lgs_dich = MorfeoTransmissiveElementsCatalog.lgs_dichroic_005()
    wv = lgs_dich.waveset
    lma_coating = CoatingsTransmissiveElementsCatalog.lma_exp_min_001()
    substrate = GlassesTransmissiveElementsCatalog.suprasil3002_80mm_internal_001()
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_002() 
    
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=1))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGS dichroic throughput at 589 nm: %s'
          % (lgs_dich.transmittance(wv)[id_589nm]))
    print('\nCoating throughput at 589 nm: %s'
          % (lma_coating.transmittance(wv)[id_589nm]))
    print('\nSubstrate throughput at 589 nm: %s'
          % (substrate.transmittance(wv)[id_589nm]))
    print('\nAR coating throughput at 589 nm: %s'
          % (ar_coating.transmittance(wv)[id_589nm]))


def lgso_l1_throughput():
    lgso_l1 = MorfeoTransmissiveElementsCatalog.lgso_lens1_001()
    wv = lgso_l1.waveset
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
    suprasil = GlassesTransmissiveElementsCatalog.suprasil3002_108mm_internal_001()
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGSO-L1 throughput at 589 nm: %s'
          % (lgso_l1.transmittance(wv)[id_589nm]))
    print('\nAR coating throughput at 589 nm: %s'
          % (ar_coating.transmittance(wv)[id_589nm]))
    print('\nSuprasil 3002 (108 mm) throughput at 589 nm: %s'
          % (suprasil.transmittance(wv)[id_589nm]))


def lgso_l2_throughput():
    lgso_l2 = MorfeoTransmissiveElementsCatalog.lgso_lens2_001()
    wv = lgso_l2.waveset
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
    suprasil = GlassesTransmissiveElementsCatalog.suprasil3002_70mm_internal_001()
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGSO-L2 throughput at 589 nm: %s'
          % (lgso_l2.transmittance(wv)[id_589nm]))
    print('\nAR coating throughput at 589 nm: %s'
          % (ar_coating.transmittance(wv)[id_589nm]))
    print('\nSuprasil 3002 (70 mm) throughput at 589 nm: %s'
          % (suprasil.transmittance(wv)[id_589nm]))


def lgso_l3_throughput():
    lgso_l3 = MorfeoTransmissiveElementsCatalog.lgso_lens3_001()
    wv = lgso_l3.waveset
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
    suprasil = GlassesTransmissiveElementsCatalog.suprasil3002_40mm_internal_001()
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGSO-L2 throughput at 589 nm: %s'
          % (lgso_l3.transmittance(wv)[id_589nm]))
    print('\nAR coating throughput at 589 nm: %s'
          % (ar_coating.transmittance(wv)[id_589nm]))
    print('\nSuprasil 3002 (40 mm) throughput at 589 nm: %s'
          % (suprasil.transmittance(wv)[id_589nm]))


def lgso_l4_throughput():
    lgso_l4 = MorfeoTransmissiveElementsCatalog.lgso_lens4_001()
    wv = lgso_l4.waveset
    ar_coating = CoatingsTransmissiveElementsCatalog.ar_coating_589nm_001()
    suprasil = GlassesTransmissiveElementsCatalog.suprasil3002_60mm_internal_001()
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGSO-L2 throughput at 589 nm: %s'
          % (lgso_l4.transmittance(wv)[id_589nm]))
    print('\nAR coating throughput at 589 nm: %s'
          % (ar_coating.transmittance(wv)[id_589nm]))
    print('\nSuprasil 3002 (60 mm) throughput at 589 nm: %s'
          % (suprasil.transmittance(wv)[id_589nm]))


def lgso_fm_throughput():
    lgso_fm = MorfeoTransmissiveElementsCatalog.lgso_fm_002()
    wv = lgso_fm.waveset
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGSO-FM throughput at 589 nm: %s'
          % (lgso_fm.reflectance(wv)[id_589nm]))


def lgs_wfs_throughput():
    lgs_wfs = MorfeoTransmissiveElementsCatalog.lgs_wfs_001()
    wv = lgs_wfs.waveset
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][0]
                                    
    print(wv[id_589nm])
    print('\nLGS WFS throughput at 589 nm: %s'
          % (lgs_wfs.transmittance(wv)[id_589nm]))


def c_blue_qe():
    c_blue = DetectorsTransmissiveElementsCatalog.c_blue_qe_001()
    wv = c_blue.waveset
    id_589nm = np.where(np.isclose(np.array(wv), WV_589NM.to(u.Angstrom).value,
                                    atol=10))[0][1]
                                    
    print(wv[id_589nm])
    print('\nC-BLUE QE at 589 nm: %s'
          % (c_blue.transmittance(wv)[id_589nm]))
