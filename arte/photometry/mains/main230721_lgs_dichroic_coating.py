import astropy.units as u
import matplotlib.pyplot as plt
from arte.photometry.transmissive_elements_catalogs import CoatingsCatalog


def main230721_comparison_of_LMA_coating_curves():
    '''
    '''
    exp_ave = CoatingsCatalog.lma_exp_ave_001()
    exp_min = CoatingsCatalog.lma_exp_min_001()
    env_min = CoatingsCatalog.lma_env_min_001()
    
    wv = exp_ave.waveset
    plt.figure()
    plt.plot(wv.to(u.um), env_min.transmittance(wv), label='env min')
    plt.plot(wv.to(u.um), exp_min.transmittance(wv), label='exp min')
    plt.plot(wv.to(u.um), exp_ave.transmittance(wv), label='exp ave')
    plt.grid()
    plt.legend()
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Transmittance')
    
    plt.figure()
    plt.plot(wv.to(u.um), env_min.reflectance(wv), label='env min')
    plt.plot(wv.to(u.um), exp_min.reflectance(wv), label='exp min')
    plt.plot(wv.to(u.um), exp_ave.reflectance(wv), label='exp ave')
    plt.grid()
    plt.legend()
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Reflectance')
