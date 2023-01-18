import numpy as np
from astropy import units as u
from synphot.spectrum import SpectralElement
from synphot.models import Empirical1D
from synphot.utils import generate_wavelengths, merge_wavelengths
from astropy.io import fits
from arte.utils.package_data import dataRootDir
import os


def standard_waveset():
    _MIN_LAMBDA_NM = 200
    _MAX_LAMBDA_NM = 10000
    _DELTA_LAMBDA_NM = 10
    return generate_wavelengths(
        minwave=_MIN_LAMBDA_NM,
        maxwave=_MAX_LAMBDA_NM,
        delta=_DELTA_LAMBDA_NM,
        log=False, wave_unit=u.nm)[0].to(u.angstrom)


class Bandpass():

    @classmethod
    def one(cls):
        return cls.flat(1)

    @classmethod
    def zero(cls):
        return cls.flat(0)

    @classmethod
    def flat(cls, amplitude):
        wv = standard_waveset()
        return SpectralElement(
            Empirical1D, points=wv, lookup_table=amplitude * np.ones(wv.shape))

    @classmethod
    def from_array(cls, waveset, values):
        return SpectralElement(Empirical1D, points=waveset, lookup_table=values)

    @classmethod
    def ramp(cls, low_wl, low_ampl, high_wl, high_ampl):
        '''
        T=low_ampl for lambda < low_wl
        T=high_ampl for lambda > high_wl
        T=linear_interpolation(low_ampl, high_ampl) for low_wl < lambda < high_wl
        '''
        l1 = low_wl.to(u.nm).value
        l2 = high_wl.to(u.nm).value
        return SpectralElement(
            Empirical1D,
            points=np.array([cls._MIN_LAMBDA_NM, l1, l2, cls._MAX_LAMBDA_NM]
                            ) * u.nm,
            lookup_table=np.array([low_ampl, low_ampl, high_ampl, high_ampl]))

    @classmethod
    def step(cls, wl, low_ampl, high_ampl):
        '''
        T=low_ampl for lambda < wl
        T=high_ampl for lambda > wl
        '''
        return cls.ramp(wl, low_ampl, wl + 1e-8 * u.Angstrom, high_ampl)


class TransmissiveElement():
    '''
    An optical element with transmittance, reflectance and absorptance

    Only 2 of the 3 parameters must be specified, as the sum of the 3 quantities
    must be equal to 1 by energy conservation at every wavelength.

    Absorptance and absorptivity are synonimous. The emissivity is equal to the
    absorptance under the assumptions of Kirchhoff's law of thermal radiation.
    (Absorpance is instead a quantity related to the optical depth).

    When a scalar value is specified, a constant spectrum of the specified
    amplitude is assumed

    Parameters
    ----------
        transmittance: `synphot.SpectralElement` or scalar in [0,1]
            spectral transmittance of the element (in wavelength)

        reflectance: `synphot.SpectralElement` or scalar in [0,1]
            spectral reflectance of the element (in wavelength)

        absorptance: `synphot.SpectralElement` or scalar in [0,1]
            spectral absorptance of the element (in wavelength)
    '''

    def __init__(self, transmittance=None, reflectance=None, absorptance=None):
        self._initialize(transmittance, reflectance, absorptance)

    def _initialize(self, transmittance=None, reflectance=None, absorptance=None):
        t, r, a = self._check_params(transmittance, reflectance, absorptance)
        self._t = t
        self._r = r
        self._a = a

    def _check_params(self, t, r, a):
        if t is None:
            if r is None or a is None:
                raise TypeError("Only one among t,r,a can be None")
            t = self._computeThird(r, a)
            return t, r, a
        if r is None:
            if t is None or a is None:
                raise TypeError("Only one among t,r,a can be None")
            r = self._computeThird(t, a)
            return t, r, a
        if a is None:
            if t is None or r is None:
                raise TypeError("Only one among t,r,a can be None")
            a = self._computeThird(t, r)
            return t, r, a
        raise TypeError("At least one among t,r,a must be None")

    def _computeThird(self, a, b):
        wv = a.waveset
        ones = Bandpass.one()
        c = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=ones(wv) - a(wv) - b(wv))
        if (np.max(c(wv)) > 1.0) or (np.min(c(wv) < 0)):
            raise ValueError("t+r+a=1 cannot be fulfilled")
        return c

    def plot(self, transmittance=True, reflectance=True,
             absorptance=True, **kwargs):
        import matplotlib.pyplot as plt
        wv = self.waveset
        if transmittance:
            plt.plot(wv, self.transmittance(wv), label='t', **kwargs)
        if reflectance:
            plt.plot(wv, self.reflectance(wv), label='r', **kwargs)
        if absorptance:
            plt.plot(wv, self.absorptance(wv), label='a', **kwargs)
        plt.legend()
        plt.grid(True)

    @property
    def waveset(self):
        return self.transmittance.waveset

    @property
    def transmittance(self):
        return self._t

    @property
    def absorptance(self):
        return self._a

    @property
    def reflectance(self):
        return self._r

    @property
    def emissivity(self):
        return self.absorptance

    def _insert_spectral_element(self, spectral_element, new_values, wavelengths):
        if new_values is None:
            return None
        ws = spectral_element.waveset
        ii = np.searchsorted(ws, wavelengths)
        new_waveset = np.insert(ws, ii, wavelengths)
        
        r_arr = spectral_element(ws)
        new_r = np.insert(r_arr, ii, new_values)
        return SpectralElement(Empirical1D,
                            points=new_waveset,
                            lookup_table=new_r)
       
    def add_spectral_points(self, wavelengths, transmittance=None,
                            reflectance=None, absorptance=None):
        new_t = self._insert_spectral_element(self.transmittance, transmittance,
                                               wavelengths)
        new_r = self._insert_spectral_element(self.reflectance, reflectance,
                                              wavelengths)
        new_a = self._insert_spectral_element(self.absorptance, absorptance,
                                              wavelengths)

        self._initialize(transmittance=new_t, reflectance=new_r,
                         absorptance=new_a)

    def to_fits(self, filename, **kwargs):
        wv = self.waveset
        hdu1 = fits.PrimaryHDU(wv.to(u.nm).value)
        hdu2 = fits.ImageHDU(self.transmittance(wv).value)
        hdu3 = fits.ImageHDU(self.absorptance(wv).value)
        hdul = fits.HDUList([hdu1, hdu2, hdu3])
        hdul.writeto(filename, **kwargs)

    @staticmethod
    def from_fits(filename):
        wv = fits.getdata(filename, 0) * u.nm
        t = SpectralElement(Empirical1D, points=wv,
                            lookup_table=fits.getdata(filename, 1))
        a = SpectralElement(Empirical1D, points=wv,
                            lookup_table=fits.getdata(filename, 2))
        return TransmissiveElement(transmittance=t, absorptance=a)


class EltTransmissiveElementsCatalog():
    
    @classmethod
    def _EltFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'elt',
            foldername)
    
    @classmethod
    def ag_mirror_elt(cls):
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('ag_mirror_elt'))
        
    @classmethod
    def al_mirror_elt(cls):
        return RestoreTransmissiveElements.restore_transmissive_elements_from_fits(
            cls._EltFolder('al_mirror_elt'))
        
    @classmethod
    def spider(cls):
        a = Bandpass.one() * 0.043
        r = Bandpass.zero()
        te = TransmissiveElement(reflectance=r, absorptance=a)
        return te


class MorfeoTransmissiveElementsCatalog():
    
    @classmethod
    def _MorfeoFolder(cls, foldername):
        return os.path.join(
            RestoreTransmissiveElements.transmissive_elements_folder(),
            'morfeo',
            foldername)
        
    @classmethod
    def lgs_dichroic(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_dichroic'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('lgs_dichroic'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        # te.add_spectral_points([200*u.nm, 7*u.um], transmittance=[0,0], reflectance=[1,1])
        return te
    
    @classmethod
    def visir_dichroic(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('visir_dichroic'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('visir_dichroic'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def infrasil_1mm(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def infrasil_1mm_B_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_B_coated'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def infrasil_1mm_C_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('infrasil_1mm_C_coated'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def lgs_lens(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('lgs_lens'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def CaF2_lens_C_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('caf2_lens_C_coated'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def adc_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('adc_coated'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('adc_coated'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def collimator_doublet_coated(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('collimator_doublet_coated'), u.um)
        r = RestoreTransmissiveElements.restore_reflectance_from_dat(
            cls._MorfeoFolder('collimator_doublet_coated'), u.um)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te
    
    @classmethod
    def notch_filter(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('notch_filter'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def ref_custom_filter(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('ref_custom_filter'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def sapphire_window(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('sapphire_window'), u.um)
        r = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, reflectance=r)
        return te

    @classmethod
    def ccd220_qe(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('ccd_220_qe'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te
    
    @classmethod
    def c_red_one_qe(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('c_red_one_qe'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

    @classmethod
    def c_red_one_filters(cls):
        t = RestoreTransmissiveElements.restore_transmittance_from_dat(
            cls._MorfeoFolder('c_red_one_filters'), u.um)
        a = Bandpass.zero()
        te = TransmissiveElement(transmittance=t, absorptance=a)
        return te

class RestoreTransmissiveElements(object):

    @classmethod
    def transmissive_elements_folder(cls):
        rootDir = dataRootDir()
        dirname = os.path.join(rootDir, 'photometry', 'optical_elements')
        return dirname

    @classmethod
    def restore_transmissive_elements_from_fits(cls, foldername):
        filename = os.path.join(foldername, '0.fits')
        return TransmissiveElement.from_fits(filename)

    @classmethod
    def restore_transmittance_from_dat(cls, foldername, wavelength_unit):
        filename = os.path.join(foldername, 't.dat')
        return SpectralElement.from_file(filename, wave_unit=wavelength_unit)

    @classmethod
    def restore_reflectance_from_dat(cls, foldername, wavelength_unit):
        filename = os.path.join(foldername, 'r.dat')
        return SpectralElement.from_file(filename, wave_unit=wavelength_unit)


class Direction(object):
    TRANSMISSION = 'transmission'
    REFLECTION = 'reflection'


class TransmissiveSystem():

    def __init__(self):
        self._elements = []
        self._waveset = np.array([1000]) * u.nm

    def add(self, transmissive_element, direction):
        self._elements.append(
            {'element': transmissive_element, 'direction': direction})
        self._waveset = merge_wavelengths(
            self._waveset.to(u.nm).value,
            transmissive_element.waveset.to(u.nm).value) * u.nm

    @property
    def transmittance(self):
        return self._compute_transmittance()
    
    def transmittance_from_to(self, from_element=0, to_element=None):
        return self._compute_transmittance(from_element, to_element)

    def _compute_transmittance(self, from_element=0, to_element=None):
        t = 1
        for el in self._elements[from_element:to_element]:
            direction = el['direction']
            if direction == Direction.TRANSMISSION:
                t *= el['element'].transmittance
            elif direction == Direction.REFLECTION:
                t *= el['element'].reflectance
            else:
                raise ValueError('Unknown propagation direction %s' % direction)
        return t

    @property
    def emissivity(self):
        return self._compute_absorptance()

    def _compute_absorptance(self):
        a = Bandpass.zero()
        for i, el in enumerate(self._elements[:-1]):
            partial_absorptance = \
                el['element'].absorptance * self._compute_transmittance(
                    from_element=i + 1)
            a = self._sum_absorptance(a, partial_absorptance)
        a = self._sum_absorptance(a, self._elements[-1]['element'].absorptance)
        return a

    def _sum_absorptance(self, a, b):
        return Bandpass.from_array(self._waveset,
                                   a(self._waveset) + b(self._waveset))

    def as_transmissive_element(self):
        return TransmissiveElement(transmittance=self.transmittance,
                                   absorptance=self.emissivity)

    def plot(self, **kwargs):
        self.as_transmissive_element().plot(reflectance=False, **kwargs)
        

def EltAsTransmissiveElement():
    spider = EltTransmissiveElementsCatalog.spider()
    m1 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m2 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m3 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m4 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m5 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    ts = TransmissiveSystem()
    ts.add(spider, Direction.TRANSMISSION)
    ts.add(m1, Direction.REFLECTION)
    ts.add(m2, Direction.REFLECTION)
    ts.add(m3, Direction.REFLECTION)
    ts.add(m4, Direction.REFLECTION)
    ts.add(m5, Direction.REFLECTION)
    return ts


def MorfeoLgsChannelAsTransmissiveElement():
#TODO: Update with current design (this comes from Cedric's spreadsheet)
    ts = EltAsTransmissiveElement()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    lgs_lens = MorfeoTransmissiveElementsCatalog.lgs_lens()
    fm = EltTransmissiveElementsCatalog.ag_mirror_elt()
    notch_filter = MorfeoTransmissiveElementsCatalog.notch_filter()
    lenslets = MorfeoTransmissiveElementsCatalog.lgs_lens() 
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window()
    ccd220 = MorfeoTransmissiveElementsCatalog.ccd220_qe()
    
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(notch_filter, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(fm, Direction.REFLECTION)
    ts.add(lenslets, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(lgs_lens, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(ccd220, Direction.TRANSMISSION)
    
    return ts


def MorfeoLowOrderChannelAsTransmissiveElement():
#TODO: Update with current design (this comes from Cedric's spreadsheet)
    ts = EltAsTransmissiveElement()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    m11up = EltTransmissiveElementsCatalog.ag_mirror_elt()
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic()
    caf2_lens = MorfeoTransmissiveElementsCatalog.CaF2_lens_C_coated()
    adc = MorfeoTransmissiveElementsCatalog.adc_coated()
    fused_silica_lenslets = MorfeoTransmissiveElementsCatalog.infrasil_1mm_C_coated()
    fused_silica_window = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    c_red1_filters = MorfeoTransmissiveElementsCatalog.c_red_one_filters()
    c_red1 = MorfeoTransmissiveElementsCatalog.c_red_one_qe()
    
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.REFLECTION)
    ts.add(caf2_lens, Direction.TRANSMISSION)
    ts.add(adc, Direction.TRANSMISSION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(fused_silica_lenslets, Direction.TRANSMISSION)
    ts.add(fused_silica_window, Direction.TRANSMISSION)
    ts.add(c_red1_filters, Direction.TRANSMISSION)
    ts.add(c_red1, Direction.TRANSMISSION)
    
    return ts


def MorfeoReferenceChannelAsTransmissiveElement():
#TODO: Update with current design (this comes from Cedric's spreadsheet)
    ts = EltAsTransmissiveElement()
    
    schmidt_plate = MorfeoTransmissiveElementsCatalog.infrasil_1mm()
    m6 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m7 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m8 = EltTransmissiveElementsCatalog.ag_mirror_elt()
    m9 = EltTransmissiveElementsCatalog.al_mirror_elt()
    m10 = EltTransmissiveElementsCatalog.al_mirror_elt()
    lgs_dichroic = MorfeoTransmissiveElementsCatalog.lgs_dichroic()
    m11up = EltTransmissiveElementsCatalog.ag_mirror_elt()
    visir_dichroic = MorfeoTransmissiveElementsCatalog.visir_dichroic()
    collimator = MorfeoTransmissiveElementsCatalog.collimator_doublet_coated()
    custom_filter = MorfeoTransmissiveElementsCatalog.ref_custom_filter()  
    fused_silica_window = MorfeoTransmissiveElementsCatalog.infrasil_1mm()   
    fused_silica_lenslets = MorfeoTransmissiveElementsCatalog.infrasil_1mm_B_coated()
    sapphire_window = MorfeoTransmissiveElementsCatalog.sapphire_window()
    ccd220 = MorfeoTransmissiveElementsCatalog.ccd220_qe()
    
    ts.add(schmidt_plate, Direction.TRANSMISSION)
    ts.add(m6, Direction.REFLECTION)
    ts.add(m7, Direction.REFLECTION)
    ts.add(m8, Direction.REFLECTION)
    ts.add(m9, Direction.REFLECTION)
    ts.add(m10, Direction.REFLECTION)
    ts.add(lgs_dichroic, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(visir_dichroic, Direction.TRANSMISSION)
    ts.add(collimator, Direction.TRANSMISSION)
    ts.add(m11up, Direction.REFLECTION)
    ts.add(custom_filter, Direction.TRANSMISSION)
    ts.add(fused_silica_window, Direction.TRANSMISSION)
    ts.add(fused_silica_lenslets, Direction.TRANSMISSION)
    ts.add(sapphire_window, Direction.TRANSMISSION)
    ts.add(ccd220, Direction.TRANSMISSION)
    
    return ts