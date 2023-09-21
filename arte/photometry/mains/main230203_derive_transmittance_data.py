import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import interpolate
from arte.photometry.transmissive_elements_catalogs import GlassesCatalog
from arte.photometry.transmittance_calculator import internal_transmittance_calculator, \
    attenuation_coefficient_calculator, external_transmittance_calculator


def main230620_get_dichroic_reflectance_from_LAM_data_AOI_11dot3deg():
    r_abi = np.loadtxt(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/From_AndreaBianco/Dichroic_MORFEO_reflectance_design_march22v3.txt')
    env_min = np.stack((r_abi[:, 0] * 1e-3, r_abi[:, 5] * 1e-2), axis=1)
    exp_min = np.stack((r_abi[:, 0] * 1e-3, r_abi[:, 6] * 1e-2), axis=1)
    exp_ave = np.stack((r_abi[:, 0] * 1e-3, r_abi[:, 7] * 1e-2), axis=1)
    np.savetxt(
        '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/lgs_dichroic_lma_env_min_aoi_11.3deg_001/r.dat',
        env_min)
    np.savetxt(
        '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/lgs_dichroic_lma_exp_min_aoi_11.3deg_001/r.dat',
        exp_min)
    np.savetxt(
        '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/lgs_dichroic_lma_exp_ave_aoi_11.3deg_001/r.dat',
        exp_ave)


def main230620_get_dichroic_transmittance_from_LAM_data_AOI_11dot3deg():
    t_abi = np.loadtxt(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Throughput_BUTTA/Data/Throughput/From_AndreaBianco/Dichroic_MORFEO_transmittancedesign_march22v3.txt')
    env_min = np.stack((t_abi[:, 0] * 1e-3, t_abi[:, 5] * 1e-2), axis=1)
    exp_min = np.stack((t_abi[:, 0] * 1e-3, t_abi[:, 6] * 1e-2), axis=1)
    exp_ave = np.stack((t_abi[:, 0] * 1e-3, t_abi[:, 7] * 1e-2), axis=1)
    
    # Add dummy data to be consistent with the wavelength range of reflectance data:
    wvset1 = np.arange(0.400, 0.580, 0.001)
    wvset2 = np.arange(0.601, 2.501, 0.001)
    data1 = np.zeros(wvset1.shape)
    data2 = np.zeros(wvset2.shape)
    
    env_min_final = np.vstack((
        np.vstack(
            (np.stack((wvset1, data1), axis=1), env_min)),
        np.stack((wvset2, data2), axis=1)
        ))
    exp_min_final = np.vstack((
        np.vstack(
            (np.stack((wvset1, data1), axis=1), exp_min)),
        np.stack((wvset2, data2), axis=1)
        ))
    exp_ave_final = np.vstack((
        np.vstack(
            (np.stack((wvset1, data1), axis=1), exp_ave)),
        np.stack((wvset2, data2), axis=1)
        ))
    
    np.savetxt(
        '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/lgs_dichroic_lma_env_min_aoi_11.3deg_001/t.dat',
        env_min_final)
    np.savetxt(
        '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/lgs_dichroic_lma_exp_min_aoi_11.3deg_001/t.dat',
        exp_min_final)
    np.savetxt(
        '/Users/giuliacarla/git/arte/arte/data/photometry/transmissive_elements/coatings/lgs_dichroic_lma_exp_ave_aoi_11.3deg_001/t.dat',
        exp_ave_final)
    
    
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
    pbl_10mm = GlassesCatalog.ohara_PBL35Y_10mm_internal_001()
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
    sk1300_10mm = GlassesCatalog.ohara_quartz_SK1300_10mm_internal_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_001()
    supra85 = GlassesCatalog.suprasil3002_85mm_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_001()
    supra85 = GlassesCatalog.suprasil3002_85mm_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_001()
    supra85 = GlassesCatalog.suprasil3002_85mm_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_001()
    supra85 = GlassesCatalog.suprasil3002_85mm_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_001()
    supra85 = GlassesCatalog.suprasil3002_85mm_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_001()
    supra85 = GlassesCatalog.suprasil3002_85mm_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_internal_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_internal_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_internal_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_internal_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_internal_001()
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
    supra10 = GlassesCatalog.suprasil3002_10mm_internal_001()
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
