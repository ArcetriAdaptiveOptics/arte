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
            _ = multi1.ensemble_average()

        _ = multi2.ensemble_average()  # Does not raise

