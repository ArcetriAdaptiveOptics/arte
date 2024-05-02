# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.dataelab.analyzer_plots import modalplot


class AnalyzerPlotsTest(unittest.TestCase):
    def test_modalplot(self):
        _ = modalplot(np.arange(10), np.arange(10)*2, title='Modaplot')


