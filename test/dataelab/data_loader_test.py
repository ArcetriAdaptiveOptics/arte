import os
import numpy as np
import unittest
from pathlib import Path
from arte.dataelab.data_loader import FitsDataLoader, NumpyDataLoader, data_loader_factory

class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        mydir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.fitsfile = os.path.join(mydir, 'testdata', 'range2x3.fits')
        self.npyfile = os.path.join(mydir, 'testdata', 'range2x3.npy')
        self.npzfile = os.path.join(mydir, 'testdata', 'range2x3.npz')
        self.npzpath = Path(mydir) / 'testdata' / 'range2x3.npz'
        self.fitsfile3d = os.path.join(mydir, 'testdata', 'range3x3x2.fits')
        self.testdata = np.arange(6).reshape((2, 3))

    def test_filename(self):
        loader = FitsDataLoader(self.fitsfile)
        assert loader.filename() == self.fitsfile

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

    def test_accept_path(self):
        loader = NumpyDataLoader(self.npzpath)
        np.testing.assert_array_equal(loader.load(), self.testdata)

    def test_axes_transpose(self):
        loader = FitsDataLoader(self.fitsfile3d, transpose_axes=(2,0,1))
        np.testing.assert_almost_equal(loader.load(), np.arange(18).reshape(2,3,3))

    def test_guess_fits(self):
        loader = data_loader_factory('foo.fits')
        assert isinstance(loader, FitsDataLoader)

    def test_guess_npy(self):
        loader = data_loader_factory('foo.npy')
        assert isinstance(loader, NumpyDataLoader)

    def test_guess_npz(self):
        loader = data_loader_factory('foo.npz')
        assert isinstance(loader, NumpyDataLoader)

    def test_guess_unknown(self):
        with self.assertRaises(ValueError):
            _ = data_loader_factory('foo.bar')
            
    def test_guess_pathlib(self):
        loader = data_loader_factory(Path('foo.npz'))
        assert isinstance(loader, NumpyDataLoader)

if __name__ == "__main__":
    unittest.main()
