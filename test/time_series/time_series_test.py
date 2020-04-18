#!/usr/bin/env python
import unittest
import math
import numpy as np
import astropy.units as u
from arte.time_series import TimeSeries, TimeSeriesWithInterpolation
from arte.time_series import Indexer, ModeIndexer


class ATimeSeries(TimeSeries):

    def __init__(self, samplingTime):
        TimeSeries.__init__(self, samplingTime)
        self.freq = 1. / samplingTime
        self.N_MODES = 6
        self.indexer = ModeIndexer(max_mode=self.N_MODES)

    def get_data(self):
        return self._get_not_indexed_data()

    def _get_not_indexed_data(self):
        dataArray = self._create_6_sine()
        return dataArray

    def get_index_of(self, *args, **kwargs):
        return np.intersect1d(
            self.indexer.modes(*args, **kwargs),
            np.arange(self.N_MODES))

    def _create_6_sine(self):
        self.disturbFreq = np.arange(1, self.N_MODES + 1) * 10.
        ret = np.zeros((1000, self.N_MODES))
        for i in np.arange(self.N_MODES):
            ret[:, i] = self._sine(self.disturbFreq[i], 1., self.freq.value) + i
        return ret

    @staticmethod
    def _sine(frequency, length, rate):
        length = int(length * rate)
        factor = np.float(frequency) * (math.pi * 2) / rate
        return np.sin(np.arange(length) * factor)

class AnIncompleteTimeSeries(TimeSeriesWithInterpolation):
    
    test_data = np.arange(16).reshape((4, 4))
    test_counter = np.array([0,1,3,4], dtype=np.uint32)

    def _get_not_indexed_data(self):
        data = self.test_data
        return self.interpolate_missing_data(data)
    
    def _get_counter(self):
        return self.test_counter

    def get_index_of(self, *args, **kwargs):
        return Indexer.myrange(*args, maxrange=4)
        

class TimeSeriesTest(unittest.TestCase):

    def setUp(self):
        dt = 0.001 * u.s
        self._ts = ATimeSeries(dt)

    def test_power_return_correct_freq(self):
        power = self._ts.power()
        freq = self._ts.frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 0])], 10.)

    def test_ensemble_size(self):
        self.assertEqual(self._ts.ensemble_size(), 6)

    def test_power_return_with_modes(self):
        power = self._ts.power(modes=4)
        freq = self._ts.frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 0])],
                               self._ts.disturbFreq[4])

    def test_power_with_from_to(self):
        power = self._ts.power(from_freq=30, to_freq=100)
        freq = self._ts.last_cutted_frequency()
        self.assertAlmostEqual(freq[np.argmax(power[:, 3])],
                               self._ts.disturbFreq[3])
        self.assertAlmostEqual(np.min(freq), 30)
        self.assertAlmostEqual(np.max(freq), 100)

    def test_timeAverage(self):
        ta = self._ts.time_average()
        self.assertAlmostEqual(ta[1], 1.)

    def test_timeAverage_withTimes(self):
        ta = self._ts.time_average(times=[0.1, 0.3])
        self.assertAlmostEqual(ta[1], 1.)

    def test_interpolation(self):
        series = AnIncompleteTimeSeries(samplingInterval=1)
        
        origin_counter = series.get_original_counter()
        interp_counter = series.get_counter()
        data = series.get_data()
        
        ref_origin = np.array([0,1,3,4], dtype=np.uint32)
        ref_interp = np.array([0,1,2,3,4], dtype=np.uint32)
        ref_data   = np.array([6,7,8,9], dtype=np.float32)
        
        np.testing.assert_array_equal(origin_counter, ref_origin)
        np.testing.assert_array_equal(interp_counter, ref_interp)
        np.testing.assert_array_equal(data[2,:], ref_data)

        assert data.shape == (5,4)

if __name__ == "__main__":
    unittest.main()
