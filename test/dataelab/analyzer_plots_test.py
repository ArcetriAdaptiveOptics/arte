# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np
from astropy.units import u

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.dataelab.analyzer_plots import modalplot


class AnalyzerPlotsTest(unittest.TestCase):
    def test_modalplot(self):
        _ = modalplot(np.arange(10), np.arange(10)*2, title='Modaplot')

    def test_modalplot_no_title(self):
        with self.assertRaises(ValueError):
            _ = modalplot(np.arange(10), np.arange(10)*2)

    def test_overplot(self):
        _ = modalplot(np.arange(10), np.arange(10)*2, overplot=True)

    def test_overplot_units(self):
        _ = modalplot(np.arange(10), np.arange(10)*2, overplot=True, unit=u.m)
