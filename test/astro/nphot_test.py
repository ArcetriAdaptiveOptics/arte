
import math
import unittest

from arte.astro.n_phot import n_phot

class NPhotTest(unittest.TestCase):

    def test_nphot1(self):
        '''
        compute the number of photons coming from a 5-mag star
        observed in V-band, with 8m-diameter telescope and an integration
        time of 12s:
        '''
        nb_of_photons, background = n_phot(5.,
                                           band='V',
                                           surf=math.pi*(8.**2)/4,
                                           delta_t=12.)

        self.assertAlmostEqual(nb_of_photons, 58226029796.93801)
        self.assertAlmostEqual(background, 23180.19997502259)


    def test_nphot2(self):
        '''
        compute the same stuff but with a user-defined sky background
        of 19.5-mag:
        '''
        nb_of_photons, background = n_phot(5.,
                                           band='V',
                                           surf=math.pi*(8.**2)/4,
                                           delta_t=12.,
                                           back_mag=19.5)

        self.assertAlmostEqual(nb_of_photons, 58226029796.93801)
        self.assertAlmostEqual(background, 92282.03824920504)


    def test_nphot3(self):
        ''' 
        compute the same stuff but with a user-defined band of
        central wavelength 0.54um and a narrow bandwidth of 0.01um:
        '''
        nb_of_photons, background = n_phot(5.,
                                           lambda_=5.4E-7,
                                           width=1E-8,
                                           surf=math.pi*(8.**2)/4,
                                           delta_t=12.,
                                           back_mag=19.5)

        self.assertAlmostEqual(nb_of_photons, 6423300529.182132)
        self.assertAlmostEqual(background, 10180.245281832633)

    def test_model(self):

        model = n_phot.get_model()
        model = n_phot.get_model('MAORY-1')

        with self.assertRaises(Exception):
            model = n_phot.get_model('foo')



