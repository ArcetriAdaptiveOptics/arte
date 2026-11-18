# -*- coding: utf-8 -*-

import gc
import weakref
import tempfile
import unittest

import numpy as np

from arte.utils.decorator import cacheResult
from arte.dataelab.cache_on_disk import cache_on_disk, set_tag, set_tmpdir
from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.types.mask import CircularMask
from arte.types.wavefront import Wavefront
from arte.utils.zernike_decomposer import ZernikeModalDecomposer


class Counter():
    def __init__(self):
        self.calls = 0

    @cacheResult
    def add(self, a, b=0):
        self.calls += 1
        return a + b


class DiskCached():
    def __init__(self, tag, tmpdir):
        set_tag(self, tag)
        set_tmpdir(self, tmpdir)

    @cache_on_disk
    def data(self):
        return np.arange(10)


class LifetimeAnalyzer(BaseAnalyzer):
    pass


def _is_freed(ref):
    gc.collect()
    return ref() is None


class CacheResultTest(unittest.TestCase):

    def test_results_are_cached(self):
        c = Counter()
        assert c.add(1, b=2) == 3
        assert c.add(1, b=2) == 3
        assert c.calls == 1
        assert c.add(1) == 1
        assert c.add(1, 2) == 3
        assert c.calls == 3

    def test_array_arguments(self):
        c = Counter()
        c.add(np.zeros(4))
        c.add(np.zeros(4))
        assert c.calls == 1
        assert c.add(np.zeros((2, 2))).shape == (2, 2)
        assert c.add(np.zeros(4, dtype=np.float32)).dtype == np.float32
        assert c.calls == 3

    def test_caches_are_separate(self):
        c1 = Counter()
        c2 = Counter()
        c1.add(1)
        c2.add(1)
        assert c1.calls == 1
        assert c2.calls == 1


class CacheLifetimeTest(unittest.TestCase):
    '''Caches must not keep alive the objects that used them'''

    def test_cacheResult(self):
        c = Counter()
        c.add(1)
        ref = weakref.ref(c)
        del c
        assert _is_freed(ref)

    def test_timeseries(self):
        series = BaseTimeSeries(np.arange(12).reshape(4, 3))
        series.get_data()
        series.get_time_vector()
        series.ensemble_size()
        series.time_size()
        ref = weakref.ref(series)
        del series
        assert _is_freed(ref)

    def test_modal_decomposer(self):
        decomposer = ZernikeModalDecomposer(n_modes=3)
        mask = CircularMask((32, 32), 10, (16, 16))
        decomposer.measureZernikeCoefficientsFromWavefront(
            Wavefront(np.ones((32, 32))), mask)
        ref = weakref.ref(decomposer)
        del decomposer
        assert _is_freed(ref)

    def test_cache_on_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            obj = DiskCached('lifetime_test', tmpdir)
            obj.data()
            ref = weakref.ref(obj)
            del obj
            assert _is_freed(ref)

    def test_analyzer_get(self):
        analyzer = LifetimeAnalyzer.get('20240404_024500')
        ref = weakref.ref(analyzer)
        del analyzer
        LifetimeAnalyzer.clear_get_cache()
        assert _is_freed(ref)


if __name__ == "__main__":
    unittest.main()
