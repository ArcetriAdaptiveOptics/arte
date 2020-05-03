# -*- coding: utf-8 -*-

import os
import unittest
from astropy.io import fits
from unittest.mock import patch

from arte.utils.decorator import cache_on_disk


class FitsStorage:
    storage = None
    filename = None

    @classmethod
    def __call__(cls, fname, data):
        cls.filename = fname
        cls.storage = data


class DecoratorTest(unittest.TestCase):

    @cache_on_disk(fname='pippo')
    def long_calculation():
        return 1

    def test_fits_cache_hit(self):

        with patch.object(os.path, 'exists', lambda fname: True):
            with patch.object(fits, 'getdata', lambda fname: 2):
                a = DecoratorTest.long_calculation()

        assert a == 2

    def test_fits_cache_miss(self):

        with patch.object(os.path, 'exists', lambda fname: False):
            with patch.object(fits, 'writeto', FitsStorage()):
                a = DecoratorTest.long_calculation()

        assert a == 1
        assert FitsStorage.storage == 1
        assert FitsStorage.filename == 'pippo'
