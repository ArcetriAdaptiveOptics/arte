# -*- coding: utf-8 -*-

import os
import numpy as np
import unittest

from arte.dataelab.data_loader import FitsDataLoader, NumpyDataLoader

class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range2x3.fits')
        self.npyfile = os.path.join(mydir, 'testdata', 'range2x3.npy')
        self.npzfile = os.path.join(mydir, 'testdata', 'range2x3.npz')
        self.testdata = np.arange(6).reshape((2, 3))

    def test_fits_assert_exists(self):
        loader = FitsDataLoader(self.fitsfile)
        loader.assert_exists()

    def test_numpy_assert_exists(self):
        myfullpath = os.path.abspath(__file__)
        loader = NumpyDataLoader(self.npyfile)
        loader.assert_exists()

    def test_fits_assert_not_exists(self):
        fullpath = '/tmp/file_does_not_exists.fits'
        loader = FitsDataLoader(fullpath)
        with self.assertRaises(AssertionError):
            loader.assert_exists()

    def test_numpy_assert_not_exists(self):
        fullpath = '/tmp/file_does_not_exists.fits'
        loader = NumpyDataLoader(fullpath)
        with self.assertRaises(AssertionError):
            loader.assert_exists()

    def test_fits_load(self):
        loader = FitsDataLoader(self.fitsfile)
        np.testing.assert_array_equal(loader.load(), self.testdata)

    def test_npy_load(self):
        loader = NumpyDataLoader(self.npyfile)
        np.testing.assert_array_equal(loader.load(), self.testdata)

    def test_npz_load(self):
        loader = NumpyDataLoader(self.npzfile)
        np.testing.assert_array_equal(loader.load(), self.testdata)

