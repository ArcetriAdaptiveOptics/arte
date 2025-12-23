from math import e
import warnings
import numpy as np
from astropy import units as u
from synphot.spectrum import SpectralElement
from synphot.models import Empirical1D
from synphot.utils import generate_wavelengths, merge_wavelengths
from astropy.io import fits


def set_element_id_from_method(func):
    """
    Decorator that sets the 'id' attribute of a returned TransmissiveElement
    to the name of the method.
    """
    def wrapper(cls, *args, **kwargs):
        element = func(cls, *args, **kwargs)
        if isinstance(element, TransmissiveElement):
            element.set_id(func.__name__)
        return element
    return wrapper


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

    def __init__(self, transmittance=None, reflectance=None, absorptance=None, id=""):
        self._initialize(transmittance, reflectance, absorptance)
        self._id = id

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
        lookup_values = ones(wv) - a(wv) - b(wv)
        
        # Check in range [0,1]
        min_val = np.min(lookup_values)
        max_val = np.max(lookup_values)
        
        NUMERICAL_ERROR_THRESHOLD = 1e-10
        
        if min_val < -NUMERICAL_ERROR_THRESHOLD:
            warnings.warn(
                f"t+r+a=1 cannot be fulfilled: minimum value is {min_val:.2e} "
                f"(threshold: {-NUMERICAL_ERROR_THRESHOLD:.2e})")
        
        if max_val > (1.0 + NUMERICAL_ERROR_THRESHOLD):
            warnings.warn(
                f"t+r+a=1 cannot be fulfilled: maximum value is {max_val:.2e} "
                f"(threshold: {1.0 + NUMERICAL_ERROR_THRESHOLD:.2e})")
        
        # If we reach here, the errors are numerical: clip the values
        lookup_values = np.clip(lookup_values, 0, 1)
        
        c = SpectralElement(
            Empirical1D, points=wv,
            lookup_table=lookup_values)
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
    def waverange(self):
        return self.transmittance.waverange

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

    @property
    def id(self):
        return self._id

    def set_id(self, id):
        self._id = id

    def _compute_average_in_band(self, spectral_el, wv_band, atol, ext_waveset):
        """
        Compute average of spectral element in a wavelength band using linear interpolation.
        
        This method interpolates the spectral element at the exact band limits
        rather than searching for nearest wavelengths, eliminating the need for atol.
        
        Parameters
        ----------
        spectral_el : SpectralElement
            The spectral element to average
        wv_band : tuple of Quantity
            (wv_min, wv_max) wavelength band limits
        atol : tuple
            Ignored in this implementation (kept for backward compatibility)
        ext_waveset : array-like or None
            Optional external waveset for integration
            
        Returns
        -------
        wv_min, wv_max, average : tuple
            The band limits and the average value in the band
            
        Risks
        -----
        - Extrapolation: If band limits are outside the spectral element's waveset,
          synphot will extrapolate (usually as constant). Check that wv_band is
          within the valid range of the spectral element.
        - Interpolation accuracy: Linear interpolation may not accurately represent
          sharp features in the spectrum. For spectral elements with high-frequency
          variations, ensure adequate sampling.
        """
        if ext_waveset is None:
            wv = self.waveset
        else:
            wv = ext_waveset
        
        wv_min = wv_band[0]
        wv_max = wv_band[1]
        
        # Ensure both are in the same unit
        wv_min = wv_min.to(wv.unit)
        wv_max = wv_max.to(wv.unit)
        
        # Check if band limits are within waveset bounds
        if wv_min < np.min(wv) or wv_max > np.max(wv):
            warnings.warn(
                f"Band limits [{wv_min}, {wv_max}] extend beyond waveset range "
                f"[{np.min(wv)}, {np.max(wv)}]. Extrapolation will be used. "
                f"Element id: {self.id}")
        
        # Generate dense waveset within the band for accurate integration
        # Use the original sampling if possible
        delta_wv = np.median(np.diff(wv.value))
        n_points = max(int((wv_max.value - wv_min.value) / delta_wv), 100)
        wv_band_dense = np.linspace(wv_min.value, wv_max.value, n_points) * wv.unit
        
        # Remove any potential duplicates
        wv_band_dense = np.unique(wv_band_dense)
        
        # Evaluate spectral element at dense points
        values = spectral_el(wv_band_dense)
        
        # Compute mean using trapezoidal integration
        average = np.trapezoid(values.value, wv_band_dense.to(wv.unit).value) / \
                (wv_max - wv_min).to(wv.unit).value
        
        return wv_min, wv_max, average

    def transmittance_in_band(self, wv_band, atol=(1, 1), ext_waveset=None):
        '''
        Parameters
        ----------
        wv_band: astropy.units.quantity.Quantity
            Bounds of the wavelength range where to compute the average transmittance of the TransmissiveElement.
        '''
        wv_min, wv_max, t_mean = self._compute_average_in_band(
            self.transmittance, wv_band, atol, ext_waveset)
        return t_mean, wv_min, wv_max

    def reflectance_in_band(self, wv_band, atol=(1, 1), ext_waveset=None):
        '''
        Parameters
        ----------
        wv_band: astropy.units.quantity.Quantity
            Bounds of the wavelength range where to compute the average reflectance of the TransmissiveElement.
        '''
        wv_min, wv_max, r_mean = self._compute_average_in_band(
            self.reflectance, wv_band, atol, ext_waveset)
        return r_mean, wv_min, wv_max

    def emissivity_in_band(self, wv_band, atol=(1, 1)):
        '''
        Parameters
        ----------
        wv_band: astropy.units.quantity.Quantity
            Bounds of the wavelength range where to compute the average emissivity of the TransmissiveElement.
        '''
        wv_min, wv_max, e_mean = self._compute_average_in_band(
            self.emissivity, wv_band, atol)
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

    @staticmethod
    def ideal():
        """
        Creates an ideal transmissive element with unit transmittance across all wavelengths.
        
        Returns
        -------
        TransmissiveElement
            A transmissive element with transmittance = 1.0 everywhere,
            reflectance = 0.0 everywhere, and absorptance = 0.0 everywhere.
        """
        return TransmissiveElement(transmittance=Bandpass.one(), 
                                   reflectance=Bandpass.zero())

    @staticmethod
    def flat(transmittance=1.0, reflectance=0.0):
        """
        Creates a transmissive element with constant values across all wavelengths.
        
        Parameters
        ----------
        transmittance : float, optional
            Constant transmittance value in [0,1]. Default is 1.0.
        reflectance : float, optional
            Constant reflectance value in [0,1]. Default is 0.0.
        
        Returns
        -------
        TransmissiveElement
            A transmissive element with constant transmittance and reflectance,
            and absorptance computed from t + r + a = 1.
            
        Raises
        ------
        ValueError
            If transmittance + reflectance > 1.
        """
        if transmittance + reflectance > 1.0:
            raise ValueError(
                f"transmittance ({transmittance}) + reflectance ({reflectance}) > 1.0")
        
        return TransmissiveElement(
            transmittance=Bandpass.flat(transmittance),
            reflectance=Bandpass.flat(reflectance))


class Direction(object):
    TRANSMISSION = 'transmission'
    REFLECTION = 'reflection'


class TransmissiveSystem():

    def __init__(self, name="Transmissive System"):
        self._name = name
        self._elements = []
        self._waveset = np.array([1000]) * u.nm

    def add(self, transmissive_element_or_system, direction=None, name=''):
        if transmissive_element_or_system is None:
            warnings.warn(
                "Trying to add a None element (named '%s') to TransmissiveSystem. Ignoring" % name)
            return
        if isinstance(transmissive_element_or_system, TransmissiveSystem):
            for el in transmissive_element_or_system.elements:
                self.add(el['element'], direction=el['direction'],
                         name=el['name'])
        else:
            if direction is None:
                raise ValueError(
                    "direction must be specified when adding a TransmissiveElement")
            self._elements.append(
                {'element': transmissive_element_or_system,
                 'name': name,
                 'direction': direction})
            self._waveset = merge_wavelengths(
                self._waveset.to(u.nm).value,
                transmissive_element_or_system.waveset.to(u.nm).value) * u.nm

    def remove(self, element_index):
        del self._elements[element_index]

    @property
    def name(self):
        return self._name

    @property
    def waveset(self):
        return self._waveset

    @property
    def transmittance(self):
        return self._compute_transmittance()

    def transmittance_from_to(self, from_element=0, to_element=None):
        return self._compute_transmittance(from_element, to_element)

    def subsystem_from_to(self, from_element=0, to_element=None):
        subsystem = TransmissiveSystem(name=self._name)
        if to_element is None:
            end_element = len(self._elements)
        else:
            end_element = to_element + 1
        for el in self._elements[from_element:end_element]:
            subsystem.add(el['element'], direction=el['direction'], name=el['name'])
        return subsystem
        

    def _compute_transmittance(self, from_element=0, to_element=None):
        t = 1
        if to_element is None:
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
                raise ValueError(
                    'Unknown propagation direction %s' % direction)
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
        el_idx = np.argwhere(
            list(map(lambda x: x['name'] == name, self._elements)))
        return el_idx[0][0]

    def element_from_name(self, name):
        return self._elements[self.element_idx_from_name(name)]['element']

    def as_transmissive_element(self):
        return TransmissiveElement(transmittance=self.transmittance,
                                   absorptance=self.emissivity)

    def plot(self, **kwargs):
        self.as_transmissive_element().plot(reflectance=False, **kwargs)

    def list_elements_transmittance(self, band):
        """
        Print the transmittance (or reflectance) in band for each element of the system.
        band: tuple (min, max) with band limits (units compatible with waveset)
        """
        for el in self._elements:
            name = el['name'] if el['name'] else '(no name)'
            id = el['element'].id
            direction = el['direction']
            elem = el['element']
            if direction == Direction.TRANSMISSION:
                if hasattr(elem, 'transmittance_in_band'):
                    val = elem.transmittance_in_band(band)
                    print(f"{name} ({id}) [T] {val[0]:.3f}")
                else:
                    print(f"{name} ({id}) [T] N/A")
            elif direction == Direction.REFLECTION:
                if hasattr(elem, 'reflectance_in_band'):
                    val = elem.reflectance_in_band(band)
                    print(f"{name} ({id}) [R] {val[0]:.3f}")
                else:
                    print(f"{name} ({id}) [R] N/A")
            else:
                print(f"{name} ({id}) ? N/A")
        total_val = self.total_transmittance(band)
        print(f"Total ({id}) [T] {total_val:.3f}")

    def print_elements_list(self):
        """
        Prints the list of elements with name and direction.
        """
        print(f"System: {self._name}")
        for i, el in enumerate(self._elements):
            name = el['name'] if el['name'] else '(no name)'
            id = el['element'].id
            if el['direction'] == Direction.REFLECTION:
                direction = 'R'
            elif el['direction'] == Direction.TRANSMISSION:
                direction = 'T'
            else:
                direction = '?'
            print(f"{i}: {name} ({id}) [{direction}]")
            
    def total_transmittance(self, band):
        """
        Print the total transmittance (or reflectance) in band for the system.
        band: tuple (min, max) with band limits (units compatible with waveset)
        """
        elem = self.as_transmissive_element()
        val = elem.transmittance_in_band(band) 
        return val[0]

    @staticmethod
    def combine(*transmissive_systems, name="Combined Transmissive System"):
        """
        Combine multiple TransmissiveSystems into a single system.
        
        Parameters
        ----------
        *transmissive_systems : TransmissiveSystem or list of TransmissiveSystem
            Variable number of TransmissiveSystem objects to combine, or a single list of systems.
        
        name : str, optional
            Name for the combined transmissive system. Default is "Combined Transmissive System".
        
        Returns
        -------
        TransmissiveSystem
            A new TransmissiveSystem containing all elements from the input systems.
        
        Examples
        --------
        >>> elt_ts = TransmissiveSystem("ELT")
        >>> mpo_ts = TransmissiveSystem("MPO")
        >>> # Both syntaxes work:
        >>> combined = TransmissiveSystem.combine(elt_ts, mpo_ts, name="ELT+MPO")
        >>> combined = TransmissiveSystem.combine([elt_ts, mpo_ts], name="ELT+MPO")
        """
        # Se il primo argomento è una lista/tupla, usala
        if len(transmissive_systems) == 1 and isinstance(transmissive_systems[0], (list, tuple)):
            systems = transmissive_systems[0]
        else:
            systems = transmissive_systems
        
        combined_system = TransmissiveSystem(name=name)
        for ts in systems:
            combined_system.add(ts)
        return combined_system
