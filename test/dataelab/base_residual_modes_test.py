# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np
import astropy.units as u
from arte.dataelab.base_data import BaseData
from arte.dataelab.data_loader import ConstantDataLoader
from arte.dataelab.base_slopes import BaseSlopes
from arte.dataelab.base_residual_modes import BaseResidualModes

class BaseResidualModesTest(unittest.TestCase):

    def setUp(self):
        self._data = np.arange(6).reshape((3,2))
        self._proj = np.array([[1,2], [3,4]])
        self._slopes = BaseSlopes(delta_time = 1*u.s,
                        loader = ConstantDataLoader(self._data))
        self._modalrec = BaseData(data_loader = ConstantDataLoader(self._proj))
        self._resmodes = BaseResidualModes(self._slopes,
                                           self._modalrec,
                                           astropy_unit = u.m)

    def test_projection(self):
        np.testing.assert_array_equal(self._resmodes.get_data(),
                                      (self._data @ self._proj) * self._resmodes.astropy_unit())

    def test_unit(self):
        assert self._resmodes.get_data().unit == u.m