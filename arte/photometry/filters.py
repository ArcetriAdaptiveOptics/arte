from arte.utils.package_data import dataRootDir
import os
from synphot.spectrum import SpectralElement
from astropy import units as u


class Filters(object):

    BESSEL_J = 'bessel_j'
    BESSEL_H = 'bessel_h'
    BESSEL_K = 'bessel_k'
    COUSIN_R = 'cousin_r'
    COUSIN_I = 'cousin_i'
    JOHNSON_U = 'johnson_u'
    JOHNSON_B = 'johnson_b'
    JOHNSON_V = 'johnson_v'
    JOHNSON_R = 'johnson_r'
    JOHNSON_I = 'johnson_i'
    JOHNSON_J = 'johnson_j'
    JOHNSON_K = 'johnson_k'
    ESO_ETC_U = 'eso_etc_u'
    ESO_ETC_B = 'eso_etc_b'
    ESO_ETC_V = 'eso_etc_v'
    ESO_ETC_R = 'eso_etc_r'
    ESO_ETC_I = 'eso_etc_i'
    ESO_ETC_Z = 'eso_etc_z'
    ESO_ETC_Y = 'eso_etc_y'
    ESO_ETC_J = 'eso_etc_j'
    ESO_ETC_H = 'eso_etc_h'
    ESO_ETC_K = 'eso_etc_k'
    ESO_ETC_L = 'eso_etc_l'
    ESO_ETC_M = 'eso_etc_m'
    ESO_ETC_N = 'eso_etc_n'
    ESO_ETC_Q = 'eso_etc_q'
    C_RED_ONE = 'c_red_one'
    CCD_220 = 'ccd_220'


    SYNPHOT = [
        BESSEL_J,
        BESSEL_H,
        BESSEL_K,
        COUSIN_R,
        COUSIN_I,
        JOHNSON_U,
        JOHNSON_B,
        JOHNSON_V,
        JOHNSON_R,
        JOHNSON_I,
        JOHNSON_J,
        JOHNSON_K
    ]

    ESO_ETC = [
        ESO_ETC_U,
        ESO_ETC_B,
        ESO_ETC_V,
        ESO_ETC_R,
        ESO_ETC_I,
        ESO_ETC_Z,
        ESO_ETC_Y,
        ESO_ETC_J,
        ESO_ETC_H,
        ESO_ETC_K,
        ESO_ETC_L,
        ESO_ETC_M,
        ESO_ETC_N,
        ESO_ETC_Q
    ]

    ALL = SYNPHOT+ESO_ETC

    @classmethod
    def names(cls):
        return cls.ALL

    @classmethod
    def get(cls, filter_name):
        '''
        Return synphot.SpectralElement of the specified filter
        The list of available filter is accessible via Filters.names()
        '''
        if filter_name in cls.SYNPHOT:
            return SpectralElement.from_filter(filter_name)
        else:
            method_to_call = getattr(cls, '_%s' % filter_name)
            return method_to_call()

    @staticmethod
    def _EsoEtcFiltersFolder():
        rootDir = dataRootDir()
        dirname = os.path.join(rootDir, 'photometry', 'filters', 'eso_etc')
        return dirname

    @staticmethod
    def _DetectorsFolder():
        rootDir = dataRootDir()
        dirname = os.path.join(rootDir, 'photometry', 'detectors')
        return dirname

    @classmethod
    def _eso_etc_u(cls):
        return cls._eso_etc('u', u.nm)

    @classmethod
    def _eso_etc_b(cls):
        return cls._eso_etc('b', u.nm)

    @classmethod
    def _eso_etc_v(cls):
        return cls._eso_etc('v', u.nm)

    @classmethod
    def _eso_etc_r(cls):
        return cls._eso_etc('r', u.nm)

    @classmethod
    def _eso_etc_i(cls):
        return cls._eso_etc('i', u.nm)

    @classmethod
    def _eso_etc_z(cls):
        return cls._eso_etc('z', u.um)

    @classmethod
    def _eso_etc_y(cls):
        return cls._eso_etc('y', u.nm)

    @classmethod
    def _eso_etc_j(cls):
        return cls._eso_etc('j', u.um)

    @classmethod
    def _eso_etc_h(cls):
        return cls._eso_etc('h', u.um)

    @classmethod
    def _eso_etc_k(cls):
        return cls._eso_etc('k', u.um)

    @classmethod
    def _eso_etc_l(cls):
        return cls._eso_etc('l', u.um)

    @classmethod
    def _eso_etc_m(cls):
        return cls._eso_etc('m', u.um)

    @classmethod
    def _eso_etc_n(cls):
        return cls._eso_etc('n', u.um)

    @classmethod
    def _eso_etc_q(cls):
        return cls._eso_etc('q', u.um)

    @classmethod
    def _ccd_220(cls):
        return SpectralElement.from_file(
            os.path.join(cls._DetectorsFolder(), 'ccd220.dat'),
            wave_unit=u.um)

    @classmethod
    def _c_red_one(cls):
        return SpectralElement.from_file(
            os.path.join(cls._DetectorsFolder(), 'c_red_one.dat'),
            wave_unit=u.nm)


    @classmethod
    def _eso_etc(cls, filter_name, wavelength_unit):
        return SpectralElement.from_file(
            os.path.join(cls._EsoEtcFiltersFolder(),
                         'phot_%s.dat' % filter_name),
            wave_unit=wavelength_unit)

