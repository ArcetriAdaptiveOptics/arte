#!/usr/bin/env python
import unittest
import math
import numpy as np
import astropy.units as u

from arte.time_series.time_series import TimeSeries, TimeSeriesException
from arte.time_series.indexer import ModeIndexer, RowColIndexer


class ATimeSeries(TimeSeries):

    def __init__(self, sampling_time):
        TimeSeries.__init__(self)
        self.freq = 1. / sampling_time
        self.sampling_time = sampling_time
        self.N_MODES = 6
        self.indexer = ModeIndexer(max_mode=self.N_MODES)
        self.use_units = True
        if isinstance(self.freq, u.Quantity):
            self.nsamples = int(self.freq.value)
        else:
            self.nsamples = int(self.freq)

    def _get_time_vector(self):
        return np.arange(self.nsamples) * self.sampling_time

    def _get_not_indexed_data(self):
        dataArray = self._create_6_sine()
        return dataArray

    def get_index_of(self, *args, **kwargs):
        return np.intersect1d(
            self.indexer.modes(*args, **kwargs),
            np.arange(self.N_MODES))

    def _create_6_sine(self):
        self.disturbFreq = np.arange(1, self.N_MODES + 1) * 10.
        ret = np.zeros((self.nsamples, self.N_MODES))
        f = self.freq.value if self.use_units else self.freq
        for i in np.arange(self.N_MODES):
            ret[:, i] = self._sine(self.disturbFreq[i], 1., f) + i
        return ret

    @staticmethod
    def _sine(frequency, length, rate):
        length = int(length * rate)
        factor = float(frequency) * (math.pi * 2) / rate
        return np.sin(np.arange(length) * factor)


class TimeSeries1D(TimeSeries):

    def _get_not_indexed_data(self):
        return np.arange(20).reshape(5,4)

    def get_index_of(self, *args, **kwargs):
        return args[0]   # Modified inside tests


class TimeSeries2D(TimeSeries):

    def _get_not_indexed_data(self):
        return np.arange(2*3*4).reshape(2,3,4)

    def get_index_of(self, *args, **kwargs):
        pass  # Modified inside tests   


class TimeSeriesTest(unittest.TestCase):

    def setUp(self):
        dt = 0.001 * u.s
        self._ts = ATimeSeries(dt)

        self._ts_no_units = ATimeSeries(0.001)
        self._ts_no_units.use_units = False

    def test_power_works_without_units(self):
        power = self._ts_no_units.power()
        freq = self._ts_no_units.frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 0])], 10.)
    
    def test_power_return_correct_freq(self):
        power = self._ts.power()
        freq = self._ts.frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 0])], 10.)

    def test_ensemble_size(self):
        self.assertEqual(self._ts.ensemble_size(), 6)

    def test_time_size(self):
        self.assertEqual(self._ts.time_size(), self._ts.nsamples)

    def test_power_return_with_modes(self):
        power = self._ts.power(modes=4)
        freq = self._ts.frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 0])],
                               self._ts.disturbFreq[4])

    def test_power_with_from_to(self):
        power = self._ts.power(from_freq=30, to_freq=100)
        freq = self._ts.last_cut_frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 3])],
                               self._ts.disturbFreq[3])
        self.assertAlmostEqual(np.min(freq), 30)
        self.assertAlmostEqual(np.max(freq), 100)

    def test_power_outside_range_higher(self):
        power = self._ts.power(from_freq=30, to_freq=1000)
        freq = self._ts.last_cut_frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 3])],
                               self._ts.disturbFreq[3])
        self.assertAlmostEqual(np.min(freq), 30)
        self.assertAlmostEqual(np.max(freq), 500)

    def test_power_outside_range_lower(self):
        power = self._ts.power(from_freq=0.001, to_freq=100)
        freq = self._ts.last_cut_frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 3])],
                               self._ts.disturbFreq[3])
        self.assertAlmostEqual(np.min(freq), 1)
        self.assertAlmostEqual(np.max(freq), 100)

    def test_power_outside_range_completely(self):
        power = self._ts.power(from_freq=800, to_freq=1000)
        freq = self._ts.last_cut_frequency()
        assert len(power) == 0
        assert len(freq) == 0

    def test_power_normalization_parseval(self):
        """Verify Parseval's theorem: integral of PSD equals temporal variance"""
        # Use TimeSeries1D for simpler test case
        np.random.seed(42)
        n_samples = 1000
        dt = 0.01  # seconds
        data = np.random.randn(n_samples)
        
        # Create TimeSeries
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: np.arange(n_samples) * dt
        t1d._get_not_indexed_data = lambda: data.reshape(-1, 1)
        t1d.get_index_of = lambda: None
        
        # Compute PSD (now in [unit²/Hz] after fix)
        psd = t1d.power(segment_factor=4.0)
        freq = t1d.frequency()
        df = freq[1] - freq[0]
        
        # Temporal variance (remove mean for consistency with PSD which removes DC)
        data_centered = data - np.mean(data)
        temporal_var = np.var(data_centered, ddof=0)
        
        # Spectral variance: integrate PSD over all frequencies
        # PSD is in [unit²/Hz], so sum(PSD * df) gives total variance
        spectral_var = np.sum(psd) * df
        
        # Verify Parseval's theorem: should match within ~10%
        rel_error = abs(temporal_var - spectral_var) / temporal_var
        self.assertLess(rel_error, 0.10, 
                       f"Parseval error {rel_error:.2%} exceeds 10%. "
                       f"Temporal variance: {temporal_var:.4e}, "
                       f"Spectral variance: {spectral_var:.4e}")

    def test_timeAverage(self):
        ta = self._ts.time_average()
        self.assertAlmostEqual(ta[1], 1.)

    def test_timeAverage_withTimes(self):
        ta = self._ts.time_average(times=[0.1, 0.3])
        self.assertAlmostEqual(ta[1], 1.)

    def test_ensemble_median(self):
        np.testing.assert_almost_equal(self._ts.time_median(mode=0), 0)

    def test_times_kwonly(self):
        with self.assertRaises(IndexError):
            _ = self._ts.time_average([0.1, 0.3])

    def test_1d_data(self):
        t1d = TimeSeries1D()
        assert t1d.time_size() == 5

    def test_1d_correct_indexing(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda *args, **kwargs: 1
        assert t1d.get_data().shape == (5,)

    def test_1d_correct_indexing_via_slice(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda *args, **kwargs: slice(0, 2)
        assert t1d.get_data().shape == (5, 2)

    def test_1d_correct_indexing_via_array(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda *args, **kwargs: [1, 2, 3]
        assert t1d.get_data().shape == (5, 3)

    def test_1d_wrong_indexing(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda *args, **kwargs: (1, 2)
        with self.assertRaises(IndexError):
            t1d.get_data()
    
    def test_2d_data(self):
        t2d = TimeSeries2D()
        assert t2d.time_size() == 2
        np.testing.assert_equal(t2d.time_average(), np.arange(3*4).reshape(3,4)+6)

    def test_2d_correct_indexing(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: (0, 2)
        assert t2d.get_data().shape == (2,)

    def test_2d_correct_indexing_arrays(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: ([0,1], [1,2])
        assert t2d.get_data().shape == (2, 2)
        
    def test_2d_correct_indexing_via_slice(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: (slice(1, 2), slice(3, 4))
        assert t2d.get_data().shape == (2, 1, 1)

    def test_2d_wrong_indexing(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: (1, 2, 3)
        with self.assertRaises(IndexError):
            t2d.get_data()

    def test_2d_wrong_indexing_via_slice(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: [slice(1, 2), slice(3, 4), slice(0, 2)] # 3D indexing must fail
        with self.assertRaises(IndexError):
            t2d.get_data()
            
    def test_2d_short_indexing(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: (1,)
        assert t2d.get_data().shape == (2, 4)

    def test_2d_short_indexing_via_slice(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: slice(1, 3)
        assert t2d.get_data().shape == (2, 2, 4)

    def test_2d_mean(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: None
        np.testing.assert_array_almost_equal(t2d.ensemble_average(), (5.5, 17.5))

    def test_2d_index(self):
        t2d = TimeSeries2D()
        t2d.get_index_of = lambda *args, **kwargs: RowColIndexer().rowcol(*args, **kwargs)
        test = t2d.get_data(rows=[1,2], col_from=0, col_to=3)
        assert test.shape == (2, 2, 3)    # [Time, rows, cols]

    def test_default_delta_time(self):
        t1d = TimeSeries1D()
        assert t1d.delta_time == 1

    def test_custom_regular_delta_time(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: [10, 20, 30, 40]
        assert t1d.delta_time == 10

    def test_custom_irregular_delta_time(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: [7.5, 11, 13.4, 16.7]
        self.assertAlmostEqual(t1d.delta_time, 3.3)

    def test_delta_time_unit(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: np.array([1,2,3,4]) * u.s
        assert t1d.delta_time == 1 * u.s
        
    def test_delta_time_1_sample(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: [10]
        assert t1d.delta_time == 1

    def test_delta_time_1_sample_with_unit(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: np.array([5]) * u.ms
        assert t1d.delta_time == 1 * u.ms

    def test_delta_time_no_samples(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: np.array([])
        assert t1d.delta_time == 1

    def test_invalid_time_vector(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: [1*u.s]   # Numpy cannot handle this
        with self.assertRaises(TimeSeriesException):
            _ = t1d.delta_time

    def test_wrong_time_vector_length(self):
        t1d = TimeSeries1D()
        t1d._get_time_vector = lambda: [1, 2]
        t1d._get_not_indexed_data = lambda: [1, 2, 3]
        with self.assertRaises(TimeSeriesException):
            _ = t1d.get_data(times=[0,1])

    def test_default_time_vector(self):
        t1d = TimeSeries1D()
        np.testing.assert_array_almost_equal(t1d.get_time_vector(), np.arange(5))

    def test_times_wrong_length(self):
        t1d = TimeSeries1D()
        with self.assertRaises(TimeSeriesException):
            _ = t1d.get_data(times=1)
        with self.assertRaises(TimeSeriesException):
            _ = t1d.get_data(times=[1,2,3])

    def test_times_start_only(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda: None
        assert t1d.get_data(times=[2, None]).shape == (3, 4)

    def test_times_stop_only(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda: None
        assert t1d.get_data(times=[None, 4]).shape == (4, 4)

    def test_times_bothj(self):
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda: None
        assert t1d.get_data(times=[2, 4]).shape == (2, 4)

    def test_time_ptp(self):
        """Test peak-to-peak over time dimension"""
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda: None
        # Data is [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]]
        # time_ptp should give ptp along axis 0: [16,16,16,16]
        ptp = t1d.time_ptp()
        np.testing.assert_array_equal(ptp, [16, 16, 16, 16])

    def test_ensemble_ptp(self):
        """Test peak-to-peak over ensemble (spatial) dimensions"""
        t1d = TimeSeries1D()
        t1d.get_index_of = lambda: None
        # Data is [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]]
        # ensemble_ptp should give ptp along axis 1: [3, 3, 3, 3, 3]
        ptp = t1d.ensemble_ptp()
        np.testing.assert_array_equal(ptp, [3, 3, 3, 3, 3])

    def test_time_ptp_with_masked_array(self):
        """Test time_ptp with masked arrays"""
        t1d = TimeSeries1D()
        t1d._get_not_indexed_data = lambda: np.ma.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            mask=[[0, 0, 1], [0, 0, 0], [0, 0, 0]]
        )
        t1d.get_index_of = lambda: None
        ptp = t1d.time_ptp()
        # Column 0: ptp([1, 4, 7]) = 6
        # Column 1: ptp([2, 5, 8]) = 6
        # Column 2: ptp([6, 9]) = 3 (masked value ignored)
        expected = np.ma.array([6, 6, 3])
        np.testing.assert_array_equal(ptp, expected)

    def test_ensemble_ptp_with_masked_array(self):
        """Test ensemble_ptp with masked arrays"""
        t1d = TimeSeries1D()
        t1d._get_not_indexed_data = lambda: np.ma.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            mask=[[0, 0, 1], [0, 0, 0], [0, 0, 0]]
        )
        t1d.get_index_of = lambda: None
        ptp = t1d.ensemble_ptp()
        # Row 0: ptp([1, 2]) = 1 (masked value ignored)
        # Row 1: ptp([4, 5, 6]) = 2
        # Row 2: ptp([7, 8, 9]) = 2
        expected = np.ma.array([1, 2, 2])
        np.testing.assert_array_equal(ptp, expected)
        
        
if __name__ == "__main__":
    unittest.main()
