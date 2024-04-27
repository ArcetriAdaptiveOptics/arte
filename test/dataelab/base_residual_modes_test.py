# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np
import astropy.units as u
from arte.dataelab.base_data import BaseData
from arte.dataelab.base_slopes import BaseSlopes, signal_unit
from arte.dataelab.base_residual_modes import BaseResidualModes


class BaseResidualModesTest(unittest.TestCase):

    def setUp(self):
        self._data = np.arange(6).reshape((3,2))
        self._proj = np.array([[1,2], [3,4]])
        self._slopes = BaseSlopes(1*u.s, self._data)
        self._modalrec = BaseData(self._proj)
        self._resmodes = BaseResidualModes(self._slopes,
                                           self._modalrec,
                                           astropy_unit = u.m)
        self._expected_data = (self._data @ self._proj) * self._resmodes.astropy_unit()

    def test_projection(self):
        np.testing.assert_array_equal(self._resmodes.get_data(),
                                      self._expected_data)

    def test_unit(self):
        '''Test that using assignment from unitless data works'''
        assert self._resmodes.get_data().unit == u.m

    def test_indexer(self):
        '''Test that the indexer for single modes works'''
        np.testing.assert_array_equal(self._resmodes.get_data(mode=1),
                                      self._expected_data[:,1])

    def test_unit_combination(self):
        '''Test that ResidualModes multiplies input units correctly'''
        self._data = np.arange(6).reshape((3,2)) * signal_unit
        self._proj = np.array([[1,2], [3,4]]) * (u.m / signal_unit)
        self._slopes = BaseSlopes(1*u.s, self._data)
        self._modalrec = BaseData(self._proj)
        self._resmodes = BaseResidualModes(self._slopes,
                                           self._modalrec,
                                           astropy_unit = None)
        self._expected_data = self._data.value @ self._proj.value * u.m
        np.testing.assert_array_equal(self._resmodes.get_data(mode=1),
                                      self._expected_data[:,1])

# __oOo__
