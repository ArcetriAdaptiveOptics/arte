import astropy.units as u
import os
import tempfile
import subprocess
from astropy.io import fits


class EsoSkyCalc(object):
    '''
    Interface to `EsoSkyCalc <http://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC>`_ 

    Calling sequence::

       sky = EsoSkyCalc(**kwargs)

    the full list of keywords parameters and defaults is in `the help page <http://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html>`_ 

    Examples
    --------
    Plot sky radiance for airmass=2 and full moon

    >>> sky = EsoSkyCalc(airmass=2, moon_sun_sep=180) 
    >>> plt.semilogy(sky.lam, sky.flux)

    Retrieve dictionary of default values

    >>> defdict= EsoSkyCalc.default_values()
    >>> defdict['observatory'] == '3060m'


    '''

    def __init__(self, **kwargs):
        self._defaults = EsoSkyCalc.default_values()
        for key, value in kwargs.items():
            self._defaults[key] = value
        self._create_input_file()
        self._run_cli()
        self._restore_fits()
        self._u = u.ph / u.s / u.m ** 2 / u.um / u.arcsec ** 2

    def _temp_input_file_name(self):
        return os.path.join(tempfile.gettempdir(), 'eso_sky_calc_input.txt')

    def _temp_output_file_name(self):
        return os.path.join(tempfile.gettempdir(), 'eso_sky_calc_output.fits')

    def _create_input_file(self):
        tmp_file_name = self._temp_input_file_name()
        with open(tmp_file_name, 'w') as the_file:
            for key, value in self._defaults.items():
                the_file.write('%s : %s\n' % (key, value))

    def _run_cli(self):
        res = subprocess.run([
            "skycalc_cli",
            "-i",
            self._temp_input_file_name(),
            "-o",
            self._temp_output_file_name()
            ],
            check=True)
        assert res.returncode == 0

    def _restore_fits(self):
        self._res = fits.getdata(self._temp_output_file_name())
        self._hdr = fits.getheader(self._temp_output_file_name())

    @staticmethod
    def default_values():
        dd = {}
        dd['airmass'] = 1.155
        dd['pwv_mode'] = 'season'
        dd['season'] = 0
        dd['time'] = 0
        dd['pwv'] = 2.5
        dd['msolflux'] = 130.0
        dd['incl_moon'] = 'Y'
        dd['moon_sun_sep'] = 0.0
        dd['moon_target_sep'] = 25.0
        dd['moon_alt'] = 45.0
        dd['moon_earth_dist'] = 1.0
        dd['incl_starlight'] = 'Y'
        dd['incl_zodiacal'] = 'Y'
        dd['ecl_lon'] = 135.0
        dd['ecl_lat'] = 90.0
        dd['incl_loweratm'] = 'Y'
        dd['incl_upperatm'] = 'Y'
        dd['incl_airglow'] = 'Y'
        dd['incl_therm'] = 'N'
        dd['therm_t1'] = 0.0
        dd['therm_e1'] = 0.0
        dd['therm_t2'] = 0.0
        dd['therm_e2'] = 0.0
        dd['therm_t3'] = 0.0
        dd['therm_e3'] = 0.0
        dd['vacair'] = 'vac'
        dd['wmin'] = 300.0
        dd['wmax'] = 5000.0
        dd['wgrid_mode'] = 'fixed_wavelength_step'
        dd['wdelta'] = 10
        dd['wres'] = 20000
        dd['lsf_type'] = 'none'
        dd['lsf_gauss_fwhm'] = 5.0
        dd['lsf_boxcar_fwhm'] = 5.0
        dd['observatory'] = '3060m'
        dd['temp_flag'] = 1
        return dd

    @property
    def lam(self):
        '''
        Wavelength array in astropy units of nm
        '''
        return self._res['lam'] * u.nm

    @property
    def flux(self):
        '''
        Radiance array in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux'] * self._u

    @property
    def trans(self):
        '''
        Fractional transmission
        '''
        return self._res['trans']

    @property
    def flux_sml(self):
        '''
        Scattered moonlight in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_sml'] * self._u

    @property
    def flux_ssl(self):
        '''
        Scattered starlight in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_ssl'] * self._u

    @property
    def flux_zl(self):
        '''
        Zodiacal light in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_zl'] * self._u

    @property
    def flux_tme(self):
        '''
        Molecular emission of lower atmosphere in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_tme'] * self._u

    @property
    def flux_ael(self):
        '''
        Emission lines of upper atmosphere in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_ael'] * self._u

    @property
    def flux_arc(self):
        '''
        Airglow / residual continuum in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_arc'] * self._u

    @property
    def flux_tie(self):
        '''
        Telescope / Instrument thermal emission in ph / s / m^2 / um / u.arcsec^2
        '''
        return self._res['flux_tie'] * self._u

    @property
    def trans_ma(self):
        '''
        Molecular Absorption
        '''
        return self._res['trans_ma']

    @property
    def trans_o3(self):
        '''
        Ozone UV / optical absorption
        '''
        return self._res['trans_o3']

    @property
    def trans_rs(self):
        '''
        Rayleigh scattering
        '''
        return self._res['trans_rs']

    @property
    def trans_ms(self):
        '''
        Mie scattering
        '''
        return self._res['trans_ms']
