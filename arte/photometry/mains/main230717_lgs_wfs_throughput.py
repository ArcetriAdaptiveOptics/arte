import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.morfeo_transmissive_systems import MorfeoLgsChannelTransmissiveSystem_003, \
    MorfeoLgsChannelTransmissiveSystem_005, \
    MorfeoLgsChannelTransmissiveSystem_004
from arte.photometry.transmissive_elements_catalogs import EltTransmissiveElementsCatalog, \
    MorfeoTransmissiveElementsCatalog, CoatingsTransmissiveElementsCatalog, \
    GlassesTransmissiveElementsCatalog

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
    pass


def lgso_l2_throughput():
    pass


def lgso_l3_throughput():
    pass


def lgso_l4_throughput():
    pass


def lgso_fm_throughput():
    pass


def lgs_wfs_throughput():
    pass


def c_blue_qe():
    pass
