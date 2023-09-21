import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.morfeo_transmissive_systems import MorfeoMainPathOptics_001, \
    MorfeoMainPathOptics_002, MorfeoMainPathOptics_003, MorfeoMainPathOptics_004, \
    MorfeoLGSO_001, MorfeoLGSO_002, MorfeoLGSO_003, MorfeoLGSO_004, \
    MorfeoMainPathOptics_005, MorfeoLGSO_005
from arte.photometry.transmissive_elements_catalogs import GlassesCatalog, \
    CoatingsCatalog, MorfeoTransmissiveElementsCatalog, \
    EltTransmissiveElementsCatalog


def main230623_throughput_of_CPM_with_SK1300_in_Ks():
    wv_min = 1.97 * u.um
    wv_max = 2.33 * u.um
    cpm_sk1300 = MorfeoTransmissiveElementsCatalog.schmidt_plate_003()
    waveset = cpm_sk1300.waveset
    id_min = np.where(np.isclose(np.array(waveset), wv_min.to(u.Angstrom).value,
                                 atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(waveset), wv_max.to(u.Angstrom).value,
                                 atol=50))[0][0]
    print(waveset[id_min])
    print(waveset[id_max])                             
    print('Average throughput in Ks band: %s'
           % np.mean(cpm_sk1300.transmittance(waveset)[id_min:id_max]))

    
def main230623_throughput_of_CPM_with_Suprasil3002_in_Ks():
    wv_min = 1.97 * u.um
    wv_max = 2.33 * u.um
    cpm_supra = MorfeoTransmissiveElementsCatalog.schmidt_plate_004()
    waveset = cpm_supra.waveset
    id_min = np.where(np.isclose(np.array(waveset), wv_min.to(u.Angstrom).value,
                                 atol=10))[0][0]
    id_max = np.where(np.isclose(np.array(waveset), wv_max.to(u.Angstrom).value,
                                 atol=50))[0][0]
    print(waveset[id_min])
    print(waveset[id_max])                             
    print('Average throughput in Ks band: %s'
           % np.mean(cpm_supra.transmittance(waveset)[id_min:id_max]))


def main230623_throughput_of_silver_coating_in_Ks():
    wv_min = 1.97 * u.um
    wv_max = 2.33 * u.um
    ag = EltTransmissiveElementsCatalog.ag_mirror_elt_002()
    waveset = ag.waveset
    id_min = np.where(np.isclose(np.array(waveset), wv_min.to(u.Angstrom).value,
                                 atol=1))[0][0]
    id_max = np.where(np.isclose(np.array(waveset), wv_max.to(u.Angstrom).value,
                                 atol=1))[0][0]
    print(waveset[id_min])
    print(waveset[id_max])                             
    print('Average throughput in Ks band: %s'
           % np.mean(ag.reflectance(waveset)[id_min:id_max]))


def main230623_throughput_of_aluminium_coating_in_Ks():
    wv_min = 1.97 * u.um
    wv_max = 2.33 * u.um
    al = EltTransmissiveElementsCatalog.al_mirror_elt_002()
    waveset = al.waveset
    id_min = np.where(np.isclose(np.array(waveset), wv_min.to(u.Angstrom).value,
                                 atol=100))[0][0]
    id_max = np.where(np.isclose(np.array(waveset), wv_max.to(u.Angstrom).value,
                                 atol=40))[0][0]
    print(waveset[id_min])
    print(waveset[id_max])                             
    print('Average throughput in Ks band: %s'
           % np.mean(al.reflectance(waveset)[id_min:id_max]))


def main230623_throughput_of_LGS_dichroic_coating_exp_in_Ks():
    wv_min = 1.97 * u.um
    wv_max = 2.33 * u.um
    lgs_dichroic = CoatingsCatalog.lma_exp_min_001()
    waveset = lgs_dichroic.waveset
    id_min = np.where(np.isclose(np.array(waveset), wv_min.to(u.Angstrom).value,
                                 atol=1))[0][0]
    id_max = np.where(np.isclose(np.array(waveset), wv_max.to(u.Angstrom).value,
                                 atol=1))[0][0]
    print(waveset[id_min])
    print(waveset[id_max])                             
    print('Average throughput in Ks band: %s'
           % np.mean(lgs_dichroic.reflectance(waveset)[id_min:id_max]))

    
def main230623_throughput_of_LGS_dichroic_coating_env_in_Ks():
    wv_min = 1.97 * u.um
    wv_max = 2.33 * u.um
    lgs_dichroic = CoatingsCatalog.lma_env_min_001()
    waveset = lgs_dichroic.waveset
    id_min = np.where(np.isclose(np.array(waveset), wv_min.to(u.Angstrom).value,
                                 atol=1))[0][0]
    id_max = np.where(np.isclose(np.array(waveset), wv_max.to(u.Angstrom).value,
                                 atol=1))[0][0]
    print(waveset[id_min])
    print(waveset[id_max])                             
    print('Average throughput in Ks band: %s'
           % np.mean(lgs_dichroic.reflectance(waveset)[id_min:id_max]))


def main230621_sk1300_vs_suprasil3002_10mm():
    sk1300 = GlassesCatalog.ohara_quartz_SK1300_10mm_internal_001()
    suprasil3002 = GlassesCatalog.suprasil3002_10mm_internal_001()
    wv_sk = sk1300.waveset
    wv_supra = suprasil3002.waveset
    plt.plot(wv_sk.to(u.um), sk1300.transmittance(wv_sk), label='SK-1300')
    plt.plot(wv_supra.to(u.um), suprasil3002.transmittance(wv_supra),
             label='Suprasil 3002')
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.grid()
    plt.legend()
    
    dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/ToDemetrioMagrin/'
    tosave_sk1300 = np.stack(
        (wv_sk.value * 1e-4, sk1300.transmittance(wv_sk).value), axis=1)
    np.savetxt(dirpath + 'sk1300_10mm.txt', tosave_sk1300)
    tosave_suprasil3002 = np.stack(
        (wv_supra.value * 1e-4, suprasil3002.transmittance(wv_supra).value
         ), axis=1)
    np.savetxt(dirpath + 'suprasil3002_10mm.txt', tosave_suprasil3002)


def main230621_sk1300_vs_suprasil3002():
    sk1300 = GlassesCatalog.ohara_quartz_SK1300_85mm_internal_001()
    suprasil3002 = GlassesCatalog.suprasil3002_85mm_internal_001()
    wv_sk = sk1300.waveset
    wv_supra = suprasil3002.waveset
    plt.plot(wv_sk.to(u.um), sk1300.transmittance(wv_sk), label='SK-1300')
    plt.plot(wv_supra.to(u.um), suprasil3002.transmittance(wv_supra),
             label='Suprasil 3002')
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.grid()
    plt.legend()

    
def main230621_lgs_dichroic_coating_env_min_vs_exp_min():
    env_min = CoatingsCatalog.lma_env_min_001()
    exp_min = CoatingsCatalog.lma_exp_min_001()
    wv_env = env_min.waveset
    wv_exp = exp_min.waveset
    plt.plot(wv_env.to(u.um), env_min.reflectance(wv_env), label='Env min')
    plt.plot(wv_exp.to(u.um), exp_min.reflectance(wv_exp),
             label='Exp min')
    plt.ylabel('Reflectance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.3, 2.6)
    plt.grid()
    plt.legend()
    

def main230621_mpo_v1(save=False):
    mpo = MorfeoMainPathOptics_001()
    t = mpo.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.ylim(-1e-1, 1)
    plt.grid()
    plt.title('MPO v1')
    
    if save is True:
        tosave = np.stack((wv.value * 1e-4, t(wv).value), axis=1)
        dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/ToDemetrioMagrin/'
        np.savetxt(dirpath + 'mpo_v1.txt', tosave)


def main230621_mpo_v2(save=False):
    mpo = MorfeoMainPathOptics_002()
    t = mpo.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.ylim(-1e-1, 1)
    plt.grid()
    plt.title('MPO v2')
    
    if save is True:
        tosave = np.stack((wv.value * 1e-4, t(wv).value), axis=1)
        dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/ToDemetrioMagrin/'
        np.savetxt(dirpath + 'mpo_v2.txt', tosave)

    
def main230621_mpo_v3(save=False):
    mpo = MorfeoMainPathOptics_003()
    t = mpo.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.ylim(-1e-1, 1)
    plt.grid()
    plt.title('MPO v3')
    
    if save is True:
        tosave = np.stack((wv.value * 1e-4, t(wv).value), axis=1)
        dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/ToDemetrioMagrin/'
        np.savetxt(dirpath + 'mpo_v3.txt', tosave)

    
def main230621_mpo_v4():
    mpo = MorfeoMainPathOptics_004()
    t = mpo.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.ylim(-1e-1, 1)
    plt.grid()
    plt.title('MPO v4')

    tosave = np.stack((wv.value * 1e-4, t(wv).value), axis=1)
    dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/ToDemetrioMagrin/'
    np.savetxt(dirpath + 'mpo_v4.txt', tosave)


def main230621_mpo_v5(save=False):
    mpo = MorfeoMainPathOptics_005()
    t = mpo.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.2, 2.6)
    plt.ylim(-1e-1, 1)
    plt.grid()
    plt.title('MPO v5')

    if save is True:
        tosave = np.stack((wv.value * 1e-4, t(wv).value), axis=1)
        dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/ToDemetrioMagrin/'
        np.savetxt(dirpath + 'mpo_v5.txt', tosave)

    
def main230621_lgso_v1():
    lgso = MorfeoLGSO_001()
    t = lgso.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.4, 0.8)
    plt.ylim(-5e-2, 0.7)
    plt.grid()
    plt.title('LGSO v1')
    print(t(wv).max())


def main230621_lgso_v2():
    lgso = MorfeoLGSO_002()
    t = lgso.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.4, 0.8)
    plt.ylim(-5e-2, 0.7)
    plt.grid()
    plt.title('LGSO v2')
    print(t(wv).max())

    
def main230621_lgso_v3():
    lgso = MorfeoLGSO_003()
    t = lgso.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.4, 0.8)
    plt.ylim(-5e-2, 0.7)
    plt.grid()
    plt.title('LGSO v3')
    print(t(wv).max())

    
def main230621_lgso_v4():
    lgso = MorfeoLGSO_004()
    t = lgso.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.4, 0.8)
    plt.ylim(-5e-2, 0.7)
    plt.grid()
    plt.title('LGSO v4')
    print(t(wv).max())

    
def main230621_lgso_v5():
    lgso = MorfeoLGSO_005()
    t = lgso.transmittance
    wv = t.waveset
    plt.plot(wv.to(u.um), t(wv))
    plt.ylabel('Transmittance')
    plt.xlabel('Wavelength [µm]')
    plt.xlim(0.4, 0.8)
    plt.ylim(-5e-2, 0.71)
    plt.grid()
    plt.title('LGSO v5')
    print(t(wv).max())
