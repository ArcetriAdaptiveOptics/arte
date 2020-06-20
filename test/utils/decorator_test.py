# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
from astropy.io import fits
from unittest.mock import patch

from arte.utils.decorator import cache_on_disk, cacheResult


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


class CacheDecoratorTest(unittest.TestCase):

    def test_cacheResult(self):

        class Tux(object):

            def __init__(self):
                self._nFlights = 0
                self._nRuns = 0

            @cacheResult
            def fly(self):
                self._nFlights += 1

            def nFlights(self):
                return self._nFlights

            @cacheResult
            def run(self):
                self._nRuns += 1

            def nRuns(self):
                return self._nRuns

        t = Tux()
        self.assertEqual(0, t.nFlights())
        t.fly()
        self.assertEqual(1, t.nFlights())
        t.fly()
        self.assertEqual(1, t.nFlights())

        self.assertEqual(0, t.nRuns())
        t.run()
        self.assertEqual(1, t.nRuns())

    def test_cache_result_on_arg1(self):

        class Tux(object):

            def __init__(self):
                self._nFlights = 0
                self._nLunchs = 0

            @cacheResult
            def fly(self, speedKmh):
                self._nFlights += 1
                return {'N_FLIGHTS': self._nFlights, 'SPEED': speedKmh}

            @cacheResult
            def eat(self, fish, bug):
                self._nLunchs += 1
                return {'N_LUNCHS': self._nLunchs, 'FISH': fish, 'BUG': bug}

        t = Tux()
        res = t.fly(130.)
        self.assertEqual(1, res['N_FLIGHTS'])
        self.assertEqual(130, res['SPEED'])

        res = t.fly(130.)
        self.assertEqual(1, res['N_FLIGHTS'])
        self.assertEqual(130, res['SPEED'])

        res = t.fly(150.)
        self.assertEqual(2, res['N_FLIGHTS'])
        self.assertEqual(150, res['SPEED'])

        self.assertRaises(Exception, t.fly)
        self.assertRaises(Exception, t.fly, 150., 'foo')

        res = t.eat('tuna', 'spider')
        self.assertEqual(1, res['N_LUNCHS'])
        self.assertEqual('tuna', res['FISH'])
        self.assertEqual('spider', res['BUG'])

        res = t.eat('tuna', 'spider')
        self.assertEqual(1, res['N_LUNCHS'])
        self.assertEqual('tuna', res['FISH'])
        self.assertEqual('spider', res['BUG'])

        res = t.eat('tuna', 'moths')
        self.assertEqual(2, res['N_LUNCHS'])
        self.assertEqual('tuna', res['FISH'])
        self.assertEqual('moths', res['BUG'])

        res = t.eat('salmon', 'moths')
        self.assertEqual(3, res['N_LUNCHS'])
        self.assertEqual('salmon', res['FISH'])
        self.assertEqual('moths', res['BUG'])

        res = t.eat('salmon', 'moths')
        self.assertEqual(3, res['N_LUNCHS'])

        res = t.fly((42, 3.14))
        self.assertEqual(3, res['N_FLIGHTS'])
        self.assertEqual(42, res['SPEED'][0])
        self.assertEqual(3.14, res['SPEED'][1])

        res = t.fly(np.array([42, 3.14]))
        self.assertEqual(4, res['N_FLIGHTS'])
        self.assertEqual(42, res['SPEED'][0])
        self.assertEqual(3.14, res['SPEED'][1])

        self.assertEqual("fly", t.fly.__name__)

