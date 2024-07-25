#!/usr/bin/env python
import unittest
import numpy as np
import astropy.units as u

from arte.time_series.time_series import TimeSeries
from arte.time_series.indexer import RowColIndexer
from arte.types.mask import CircularMask 


radius = 2
nsamples = 10
mask2d = CircularMask((2*radius, 2*radius), radius)
m2d = mask2d.mask()
data2d = np.arange(nsamples*4*4).reshape(nsamples, 4, 4) * u.m

data1d = np.arange(nsamples*4).reshape(nsamples, 4) * u.s
m1d = np.stack([True, True, False, True]*nsamples)


class TimeSeriesMaskedArray2D(TimeSeries):

    def _get_not_indexed_data(self):
        return np.ma.array(data2d, mask=np.stack([m2d]*nsamples))

    def get_index_of(self, *args, **kwargs):
        return None   # Modified inside tests

class TimeSeriesMaskedArray1D(TimeSeries):

    def _get_not_indexed_data(self):
        return np.ma.array(data1d, mask=m1d)

    def get_index_of(self, *args, **kwargs):
        return None   # Modified inside tests


class TimeSeriesMATest(unittest.TestCase):

    def setUp(self):
        self._ts2d = TimeSeriesMaskedArray2D()
        self._ts1d = TimeSeriesMaskedArray1D()
    
    def test_ensemble_size(self):
        self.assertEqual(self._ts2d.ensemble_size(), data2d[0].size)

    def test_time_size(self):
        self.assertEqual(self._ts2d.time_size(), len(data2d))

    def test_timeAverage(self):
        ta = self._ts2d.time_average()
        np.testing.assert_almost_equal(ta, data2d.mean(axis=0))
        np.testing.assert_array_equal(ta.mask, m2d)

    def test_timeAverage_withTimes(self):
        ta = self._ts2d.time_average(times=[1, 3])
        np.testing.assert_almost_equal(ta, (data2d[1] + data2d[2])/2)
        np.testing.assert_array_equal(ta.mask, m2d)

    def test_time_median(self):
        np.testing.assert_almost_equal(self._ts2d.time_median(mode=0).data.value, np.median(data2d, axis=0))

    def test_ensemble_std(self):
        np.testing.assert_almost_equal(self._ts2d.ensemble_std().data.value, (np.array([3.9475731]*10) * u.m).value)
        assert self._ts2d.ensemble_std().data.unit == u.m

    def test_ensemble_rms(self):
        np.testing.assert_almost_equal(self._ts2d.ensemble_rms().data.value, np.sqrt(np.mean(np.abs(data2d.data)**2, axis=(1,2))))
        assert self._ts2d.ensemble_std().data.unit == u.m

    def test_1d_correct_indexing(self):
        self._ts1d.get_index_of = lambda *args, **kwargs: 1
        assert self._ts1d.get_data().shape == (nsamples,)
        np.testing.assert_array_equal(self._ts1d.get_data().mask, [m1d[1]]*nsamples)

    def test_1d_correct_indexing_via_slice(self):
        self._ts1d.get_index_of = lambda *args, **kwargs: slice(0, 2)
        assert self._ts1d.get_data().shape == (nsamples, 2)
        np.testing.assert_array_equal(self._ts1d.get_data().mask, [m1d[0:2]] * nsamples)

    def test_1d_correct_indexing_via_array(self):
        self._ts1d.get_index_of = lambda *args, **kwargs: [1, 2, 3]
        assert self._ts1d.get_data().shape == (nsamples, 3)
        np.testing.assert_array_equal(self._ts1d.get_data().mask, [m1d[1:4]] * nsamples)

    def test_1d_wrong_indexing(self):
        self._ts1d.get_index_of = lambda *args, **kwargs: (1, 2)
        with self.assertRaises(IndexError):
            self._ts1d.get_data()
    
    def test_2d_correct_indexing(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: (0, 2)
        assert self._ts2d.get_data().shape == (nsamples,)
        np.testing.assert_array_equal(self._ts2d.get_data().mask, [m2d[0, 2]] * nsamples)

    def test_2d_correct_indexing_arrays(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: ([0,1], [1,2])
        assert self._ts2d.get_data().shape == (nsamples, 2)
        np.testing.assert_array_equal(self._ts2d.get_data().mask, [m2d[[0,1], [1,2]]] * nsamples)
        
    def test_2d_correct_indexing_via_slice(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: (slice(1, 2), slice(3, 4))
        assert self._ts2d.get_data().shape == (nsamples, 1, 1)
        np.testing.assert_array_equal(self._ts2d.get_data().mask, [m2d[1:2, 3:4]] * nsamples)

    def test_2d_wrong_indexing(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: (1, 2, 3)
        with self.assertRaises(IndexError):
            self._ts2d.get_data()

    def test_2d_wrong_indexing_via_slice(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: [slice(1, 2), slice(3, 4), slice(0, 2)] # 3D indexing must fail
        with self.assertRaises(IndexError):
            self._ts2d.get_data()
            
    def test_2d_short_indexing(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: (1,)
        assert self._ts2d.get_data().shape == (nsamples, 4)
        np.testing.assert_array_equal(self._ts2d.get_data().mask, [m2d[1]] * nsamples)

    def test_2d_short_indexing_via_slice(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: slice(0, 2)
        assert self._ts2d.get_data().shape == (nsamples, 2, 4)
        np.testing.assert_array_equal(self._ts2d.get_data().mask, [m2d[0:2]] * nsamples)

    def test_2d_mean(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: None
        np.testing.assert_array_almost_equal(self._ts2d.ensemble_average().data.value, data2d.mean(axis=(1,2)).data)

    def test_2d_index(self):
        self._ts2d.get_index_of = lambda *args, **kwargs: RowColIndexer().rowcol(*args, **kwargs)
        test = self._ts2d.get_data(rows=[1,2], col_from=0, col_to=3)
        assert test.shape == (nsamples, 2, 3)    # [Time, rows, cols]
        np.testing.assert_array_equal(test.mask, [m2d[ [1,2], 0:3 ]] * nsamples)


if __name__ == "__main__":
    unittest.main()
