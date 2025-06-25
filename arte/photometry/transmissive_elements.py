import numpy as np
from astropy import units as u
from synphot.spectrum import SpectralElement
from synphot.models import Empirical1D
from synphot.utils import generate_wavelengths, merge_wavelengths
from astropy.io import fits


class Bandpass():
    
    _MIN_LAMBDA_NM = 200
    _MAX_LAMBDA_NM = 10000
    _DELTA_LAMBDA_NM = 10

    @classmethod
    def standard_waveset(cls):
        return generate_wavelengths(
            minwave=cls._MIN_LAMBDA_NM,
            maxwave=cls._MAX_LAMBDA_NM,
            delta=cls._DELTA_LAMBDA_NM,
            log=False, wave_unit=u.nm)[0]

    @classmethod
    def one(cls):
        return cls.flat(1)

    @classmethod
    def zero(cls):
        return cls.flat(0)

    @classmethod
    def flat(cls, amplitude):
        wv = cls.standard_waveset()
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
    def peak(cls, peak_wl, delta_wl, high_ampl, low_ampl):
        '''
        T = high_ampl for lambda = peak_wl
        T = low_ampl for lambda < (peak_wl - delta_wl) and 
            lambda > (peak_wl + delta_wl)
        T = linear_interpolation(low_ampl, high_ampl) for 
            (peak_wl - delta_wl) < lambda < peak_wl and
            peak_wl < lambda < (peak_wl + delta_wl) 
        '''
        l = peak_wl.to(u.nm).value
        dl = delta_wl.to(u.nm).value
        return SpectralElement(
            Empirical1D,
            points=np.array([
                cls._MIN_LAMBDA_NM, l - dl, l, l + dl, cls._MAX_LAMBDA_NM]) * u.nm,
            lookup_table=np.array([
                low_ampl, low_ampl, high_ampl, low_ampl, low_ampl]))
 
    @classmethod
    def top_hat(cls, peak_wl, delta_wl, high_ampl, low_ampl):
        '''
        T = high_ampl for lambda = peak_wl +/- delta_wl
        T = low_ampl elsewhere
        '''
        l = peak_wl.to(u.nm).value
        dl = delta_wl.to(u.nm).value
        return SpectralElement(
            Empirical1D,
            points=np.array([
                cls._MIN_LAMBDA_NM, l - dl, l - dl + 1e-8, l + dl,
                l + dl + 1e-8, cls._MAX_LAMBDA_NM]) * u.nm,
            lookup_table=np.array([
                low_ampl, low_ampl, high_ampl, high_ampl, low_ampl, low_ampl]))

    @classmethod
    def top_hat_ramped(cls, low_wl_start, high_wl_start,
                        high_wl_end, low_wl_end,
                       low_ampl, high_ampl):
        l1 = low_wl_start.to(u.nm).value
        l2 = high_wl_start.to(u.nm).value
        l3 = high_wl_end.to(u.nm).value
        l4 = low_wl_end.to(u.nm).value 
        return SpectralElement(
            Empirical1D,
            points=np.array([cls._MIN_LAMBDA_NM, l1, l2, l3, l4, cls._MAX_LAMBDA_NM]
                            ) * u.nm,
            lookup_table=np.array([low_ampl, low_ampl, high_ampl, high_ampl, low_ampl, low_ampl])) 

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
             absorptance=True, wv_unit=None, **kwargs):
        import matplotlib.pyplot as plt
        if wv_unit is None:
            wv = self.waveset
        else:
            wv = self.waveset.to(wv_unit)
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
    
    def _compute_average_in_band(self, spectral_el, wv_band, atol, ext_waveset):
        if not ext_waveset:
            wv = self.waveset
        else:
            wv = ext_waveset
        id_wv_min = np.where(np.isclose(wv, wv_band[0], atol=atol[0]))[0]
        wv_min = wv[id_wv_min]
        if len(wv_min) == 0:
            raise ValueError('No wavelength found. Increase atol.')
        elif len(wv_min) > 1:
            diff = (wv_band[0] - wv_min).value
            if np.all(np.isclose(0, diff)) or np.isclose(abs(diff[0]), abs(diff[1])):
                wv_min = wv_min[0]
            else:
                raise ValueError(f'Too many wavelengths found: {wv_min}. Change atol.')
            
        id_wv_max = np.where(np.isclose(wv, wv_band[1], atol=atol[1]))[0]
        wv_max = wv[id_wv_max]
        if len(wv_max) == 0:
            raise ValueError('No wavelength found. Increase atol.')
        elif len(wv_max) > 1:
            diff = (wv_band[1] - wv_max).value
            if np.all(np.isclose(0, diff)) or np.isclose(abs(diff[0]), abs(diff[1])):
                wv_max = wv_max[0]
            else:
                raise ValueError(f'Too many wavelengths found: {wv_max}. Change atol.')
            
        return wv_min, wv_max, np.mean(spectral_el(wv)[id_wv_min[0]:id_wv_max[0]+1]) 

    def transmittance_in_band(self, wv_band, atol=(1, 1), ext_waveset=None):
        '''
        Parameters
        ----------
        wv_band: astropy.units.quantity.Quantity
            Bounds of the wavelength range where to compute the average transmittance of the TransmissiveElement.
        '''
        wv_min, wv_max, t_mean = self._compute_average_in_band(self.transmittance, wv_band, atol, ext_waveset)
        return t_mean, wv_min, wv_max
    
    def reflectance_in_band(self, wv_band, atol=(1, 1), ext_waveset=None):
        '''
        Parameters
        ----------
        wv_band: astropy.units.quantity.Quantity
            Bounds of the wavelength range where to compute the average reflectance of the TransmissiveElement.
        '''
        wv_min, wv_max, r_mean = self._compute_average_in_band(self.reflectance, wv_band, atol, ext_waveset)
        return r_mean, wv_min, wv_max
    
    def emissivity_in_band(self, wv_band, atol=(1, 1)):
        '''
        Parameters
        ----------
        wv_band: astropy.units.quantity.Quantity
            Bounds of the wavelength range where to compute the average emissivity of the TransmissiveElement.
        '''
        wv_min, wv_max, e_mean = self._compute_average_in_band(self.emissivity, wv_band, atol)
        print(f'Average emissivity in band {(wv_min, wv_max)}: {e_mean}')
        return e_mean

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
        
    def to_dat(self, filepath, data_type):
        if data_type == 'transmittance':
            data = self.transmittance(self.waveset).value
            title = np.array([['Wavelength [um]', 'Transmittance']])
        elif data_type == 'reflectance':
            data = self.reflectance(self.waveset).value
            title = np.array([['Wavelength [um]', 'Reflectance']])
        elif data_type == 'absorptance':
            data = self.absorptance(self.waveset).value
            title = np.array([['Wavelength [um]', 'Absorptance']])
        to_write = np.stack((
            self.waveset.to(u.um).value,
            data), axis=1)
        with open(filepath, 'a') as datafile:
            np.savetxt(datafile,
                       np.vstack((title, to_write)),
                       fmt=['%-30s', '%-30s'])

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


class Direction(object):
    TRANSMISSION = 'transmission'
    REFLECTION = 'reflection'


class TransmissiveSystem():

    def __init__(self):
        self._elements = []
        self._waveset = np.array([1000]) * u.nm

    def add(self, transmissive_element, direction, name=''):
        self._elements.append(
            {'element': transmissive_element,
             'name': name,
             'direction': direction})
        self._waveset = merge_wavelengths(
            self._waveset.to(u.nm).value,
            transmissive_element.waveset.to(u.nm).value) * u.nm

    def remove(self, element_index):
        del self._elements[element_index]

    @property
    def transmittance(self):
        return self._compute_transmittance()
    
    def transmittance_from_to(self, from_element=0, to_element=None):
        return self._compute_transmittance(from_element, to_element)

    def _compute_transmittance(self, from_element=0, to_element=None):
        t = 1
        if not to_element:
            end_element = to_element
        else:
            end_element = to_element + 1
        for el in self._elements[from_element:end_element]:
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
    
    @property
    def elements(self):
        return self._elements
    
    def element_idx_from_name(self, name):
        el_idx = np.argwhere(list(map(lambda x: x['name']==name, self._elements))) 
        return el_idx[0][0]
        # return self._elements[el_idx[0][0]]['element']
    
    def as_transmissive_element(self):
        return TransmissiveElement(transmittance=self.transmittance,
                                   absorptance=self.emissivity)

    def plot(self, **kwargs):
        self.as_transmissive_element().plot(reflectance=False, **kwargs)
        
