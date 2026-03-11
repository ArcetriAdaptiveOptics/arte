# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np
import astropy.units as u
from arte.time_series.time_series import TimeSeries
from arte.time_series.multi_time_series import MultiTimeSeries

class ATimeSeries(TimeSeries):

    def __init__(self, sampling_time):
        TimeSeries.__init__(self)
        self._sampling_time = sampling_time

    def get_data(self):
        return self._get_not_indexed_data()

    def _get_not_indexed_data(self):
        return np.arange(6).reshape((3,2))

    def _get_time_vector(self):
        return np.arange(6) * self._sampling_time

    def get_index_of(self, *args, **kwargs):
        if len(args)==0:
            return range(self.ensemble_size())
        else:
            return range(*args)

class AMultiTimeSeries(MultiTimeSeries):

    def get_index_of(self, *args, **kwargs):
        if len(args)==0:
            return range(self.ensemble_size())
        else:
            return range(*args)

class MultiTimeSeriesTest(unittest.TestCase):

    def setUp(self):
        self.series1 = ATimeSeries(1*u.s)
        self.series2 = ATimeSeries(2*u.s)
        self.series3 = ATimeSeries(3*u.s)
        self.series1b = ATimeSeries(1*u.s)

    def test_esemble_size(self):
        
        multi1 = AMultiTimeSeries(self.series1, self.series2)
        multi2 = AMultiTimeSeries(self.series1, self.series2, self.series3)

        assert multi1.ensemble_size() == 4
        assert multi2.ensemble_size() == 6

    def test_add_series(self):
        
        multi = AMultiTimeSeries(self.series1, self.series2)
        assert multi.ensemble_size() == 4

        multi.add_series(self.series3)
        assert multi.ensemble_size() == 6

    def test_is_homogeneous(self):
        
        multi1 = AMultiTimeSeries(self.series1, self.series2)
        multi2 = AMultiTimeSeries(self.series1, self.series1b)

        assert multi1.is_homogeneous() == False
        assert multi2.is_homogeneous() == True

    def test_ensemble_functions(self):
        
        multi1 = AMultiTimeSeries(self.series1, self.series2)
        multi2 = AMultiTimeSeries(self.series1, self.series1b)

        with self.assertRaises(Exception):
            _ = multi1.ensemble_mean

        _ = multi2.ensemble_mean  # Does not raise
    
    def test_ensemble_properties_chainable(self):
        """Test that ensemble properties are chainable and return TimeSeries"""
        multi = AMultiTimeSeries(self.series1, self.series1b)
        
        # ensemble_mean is chainable
        mean_series = multi.ensemble_mean
        self.assertIsInstance(mean_series, TimeSeries)
        
        # Can chain with time operations
        mean_then_time_mean = multi.ensemble_mean.time_mean
        self.assertIsInstance(mean_then_time_mean, TimeSeries)
        
        # Can extract final value
        final_value = multi.ensemble_mean.time_mean.value
        self.assertTrue(np.isscalar(final_value) or isinstance(final_value, np.ndarray))
    
    def test_all_ensemble_properties_check_homogeneity(self):
        """Test that all ensemble properties check homogeneity"""
        multi_hetero = AMultiTimeSeries(self.series1, self.series2)
        multi_homo = AMultiTimeSeries(self.series1, self.series1b)
        
        # Heterogeneous series should raise for all ensemble properties
        with self.assertRaises(Exception):
            _ = multi_hetero.ensemble_mean
        with self.assertRaises(Exception):
            _ = multi_hetero.ensemble_std
        with self.assertRaises(Exception):
            _ = multi_hetero.ensemble_median
        with self.assertRaises(Exception):
            _ = multi_hetero.ensemble_rms
        with self.assertRaises(Exception):
            _ = multi_hetero.ensemble_ptp
        
        # Homogeneous series should work for all
        _ = multi_homo.ensemble_mean
        _ = multi_homo.ensemble_std
        _ = multi_homo.ensemble_median
        _ = multi_homo.ensemble_rms
        _ = multi_homo.ensemble_ptp
    
    def test_deprecated_methods_warn(self):
        """Test that deprecated methods issue warnings"""
        multi = AMultiTimeSeries(self.series1, self.series1b)
        
        with self.assertWarns(DeprecationWarning):
            _ = multi.get_ensemble_average()
        
        with self.assertWarns(DeprecationWarning):
            _ = multi.get_ensemble_std()
        
        with self.assertWarns(DeprecationWarning):
            _ = multi.get_ensemble_median()
        
        with self.assertWarns(DeprecationWarning):
            _ = multi.get_ensemble_rms()
        
        with self.assertWarns(DeprecationWarning):
            _ = multi.get_ensemble_ptp()
    
    def test_deprecated_methods_check_homogeneity(self):
        """Test that all deprecated ensemble methods check homogeneity"""
        multi_hetero = AMultiTimeSeries(self.series1, self.series2)
        multi_homo = AMultiTimeSeries(self.series1, self.series1b)
        
        # Heterogeneous series should raise for all deprecated methods
        with self.assertRaises(Exception):
            with self.assertWarns(DeprecationWarning):
                _ = multi_hetero.get_ensemble_average()
        
        with self.assertRaises(Exception):
            with self.assertWarns(DeprecationWarning):
                _ = multi_hetero.get_ensemble_std()
        
        with self.assertRaises(Exception):
            with self.assertWarns(DeprecationWarning):
                _ = multi_hetero.get_ensemble_median()
        
        with self.assertRaises(Exception):
            with self.assertWarns(DeprecationWarning):
                _ = multi_hetero.get_ensemble_rms()
        
        with self.assertRaises(Exception):
            with self.assertWarns(DeprecationWarning):
                _ = multi_hetero.get_ensemble_ptp()
        
        # Homogeneous series should work for all (with warnings)
        with self.assertWarns(DeprecationWarning):
            _ = multi_homo.get_ensemble_average()
        with self.assertWarns(DeprecationWarning):
            _ = multi_homo.get_ensemble_std()
        with self.assertWarns(DeprecationWarning):
            _ = multi_homo.get_ensemble_median()
        with self.assertWarns(DeprecationWarning):
            _ = multi_homo.get_ensemble_rms()
        with self.assertWarns(DeprecationWarning):
            _ = multi_homo.get_ensemble_ptp()

