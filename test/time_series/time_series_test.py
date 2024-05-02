#!/usr/bin/env python
import unittest
import math
import numpy as np
import astropy.units as u
from arte.time_series import TimeSeries, TimeSeriesWithInterpolation
from arte.time_series import Indexer, ModeIndexer
from arte.utils.not_available import NotAvailable


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


class AnIncompleteTimeSeries1(TimeSeriesWithInterpolation):
    '''A time series with a single gap a'''

    test_data = np.arange(16).reshape((4, 4))   #Original data
    test_counter = np.array([0,1,3,4], dtype=np.uint32)  # Original counter

    ref_counter = np.array([0,1,2,3,4], dtype=np.uint32)
    ref_data_shape = (5,4)
    ref_data = [np.array([6,7,8,9], dtype=np.float32)]
    ref_slice = [np.s_[2,:]]  # Where to find the interpolated data
    
    def get_interpolation(self, new_data):
        return [new_data[_slice] for _slice in self.ref_slice]

    def get_original(self, new_data):
        mask = np.ones(new_data.shape,dtype=bool)
        for _slice in self.ref_slice:
             mask[_slice]= False
        return new_data[mask].reshape(self.test_data.shape)

    def _get_not_indexed_data(self):
        data = self.test_data
        return self.interpolate_missing_data(data)
    
    def _get_counter(self):
        return self.test_counter

    def get_index_of(self, *args, **kwargs):
        return Indexer.myrange(*args, maxrange=4)


class AnIncompleteTimeSeries2(AnIncompleteTimeSeries1):
    '''A time series with a single gap and a counter going up by 2'''
    
    test_data = np.arange(16).reshape((4, 4))
    test_counter = np.array([0,2,6,8], dtype=np.uint32)
    ref_counter = np.array([0,2,4,6,8], dtype=np.uint32)
    ref_data_shape = (5,4)
    ref_data = [np.array([6,7,8,9], dtype=np.float32)]
    ref_slice = [np.s_[2,:]]

class AnIncompleteTimeSeries3(AnIncompleteTimeSeries1):
    '''A time series with no gaps'''
    
    test_data = np.arange(16).reshape((4, 4))
    test_counter = np.array([0,1,2,3], dtype=np.uint32)
    ref_counter = np.array([0,1,2,3], dtype=np.uint32)
    ref_data_shape = (4,4)
    ref_data = []
    ref_slice = []

class AnIncompleteTimeSeries4(AnIncompleteTimeSeries1):
    '''A time series with three separate gaps'''

    test_data = np.arange(40).reshape((10, 4))
    test_counter = np.array([0,1,2,6,7,8,9,11,13,14], dtype=np.uint32)
    ref_counter = np.arange(15, dtype=np.uint32)
    ref_data_shape = (15,4)
    ref_data = [np.array([[9,10,11,12],
                                        [10,11,12,13],
                                        [11,12,13,14]], dtype=np.float32),
                              np.array([26,27,28,29], dtype=np.float32),
                              np.array([30,31,32,33], dtype=np.float32)]
                                       
    ref_slice = [np.s_[3:6,:], np.s_[10,:], np.s_[12,:]]


class AnIncompleteTimeSeries5(AnIncompleteTimeSeries4):
    '''A time series with three separate gaps, with a strange counter'''

    test_counter = AnIncompleteTimeSeries4.test_counter*0.1 - 10
    ref_counter = AnIncompleteTimeSeries4.ref_counter*0.1 - 10

class AnIncompleteTimeSeries6(AnIncompleteTimeSeries4):
    '''A time series with an incorrect frame counter length'''

    # Should have been 10
    test_counter = np.arange(8)

class CounterIsNotAvailable(AnIncompleteTimeSeries4):
    
    test_counter = NotAvailable()

class TimeSeriesWithInterpolationTest(unittest.TestCase):

    def _test_interpolation(self, classname):

        series = classname(sampling_interval=1)
        
        origin_counter = series.get_original_counter()
        interp_counter = series.get_counter()
        data = series.get_data()
        
        np.testing.assert_array_equal(origin_counter, series.test_counter)
        assert np.allclose(interp_counter, series.ref_counter)
        for interpolation, ref in zip(series.get_interpolation(data),
                                      series.ref_data):
            assert np.allclose(interpolation, ref)
            
        np.testing.assert_array_equal(series.get_original(data),
                                      series.test_data)

        assert data.shape == series.ref_data_shape

    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_interpolation_with_step1(self):
        self._test_interpolation(AnIncompleteTimeSeries1)

    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_interpolation_with_step2(self):
        self._test_interpolation(AnIncompleteTimeSeries2)

    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_interpolation_with_step3(self):
        self._test_interpolation(AnIncompleteTimeSeries3)

    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_interpolation_with_step4(self):
        self._test_interpolation(AnIncompleteTimeSeries4)

    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_interpolation_with_step5(self):
        self._test_interpolation(AnIncompleteTimeSeries5)

    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_interpolation_with_step6(self):
        with self.assertRaises(ValueError):
            self._test_interpolation(AnIncompleteTimeSeries6)
    
    @unittest.skip("TimeSeriesWithInterpolation must be updated")
    def test_counter_not_available(self):
        
        series = CounterIsNotAvailable(sampling_interval=1)
        
        assert isinstance(series.get_data(), NotAvailable)
        assert isinstance(series.get_original_counter(), NotAvailable)
        
if __name__ == "__main__":
    unittest.main()
