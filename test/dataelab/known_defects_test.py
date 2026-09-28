# -*- coding: utf-8 -*-
'''
Known defects of arte.dataelab, tracked in issue #78.

Each test asserts the CORRECT behaviour and is marked as an expected failure.
The identifiers (A1, B3, ...) match the checklist in the issue.

When a defect is fixed, its test starts passing and unittest reports it as an
"unexpected success", which makes the test run fail: remove the
@unittest.expectedFailure decorator in the same change that fixes the defect,
so that the test becomes a regular regression test.
'''

import os
import gc
import copy
import shutil
import pickle
import tempfile
import unittest
import weakref

import numpy as np
import astropy.units as u

import matplotlib
matplotlib.use('Agg')   # Draw in background

from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.dataelab.base_data import BaseData
from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.base_analyzer_set import BaseAnalyzerSet
from arte.dataelab.base_indexer import BaseIndexer
from arte.dataelab.cache_on_disk import cache_on_disk, set_tmpdir, get_disk_cacher, DiskCacher
from arte.dataelab.data_loader import OnTheFlyLoader, NumpyDataLoader
from arte.dataelab.tag import Tag
from arte.dataelab.analyzer_plots import modalplot


class _TmpDirTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class WrongResultsTest(_TmpDirTestCase):
    '''A: wrong results without any error'''

    @unittest.expectedFailure
    def test_A1_cache_on_disk_distinguishes_arguments(self):
        class Analyzer(BaseAnalyzer):
            @cache_on_disk
            def f(self, n):
                return np.arange(n)
        a = Analyzer('20240201_000001')
        set_tmpdir(a, self.tmpdir)
        _ = a.f(3)
        self.assertEqual(len(a.f(10)), 10)

    @unittest.expectedFailure
    def test_A2_base_indexer_custom_keywords(self):
        indexer = BaseIndexer(single_kw='mode', from_kw='from_mode', to_kw='to_mode')
        self.assertEqual(indexer.process_args(mode=5), 5)
        self.assertEqual(indexer.process_args(from_mode=2, to_mode=4), slice(2, 4))

    @unittest.expectedFailure
    def test_A2_base_indexer_custom_keywords_keep_defaults(self):
        indexer = BaseIndexer(single_kw='mode')
        self.assertEqual(indexer.process_args(element=5), 5)

    @unittest.expectedFailure
    def test_A3_recalc_refreshes_derived_series(self):
        state = {'value': 1.0}

        class Derived(BaseTimeSeries):
            def __init__(self):
                super().__init__(OnTheFlyLoader(self._calc))

            @cache_on_disk
            def _calc(self):
                return np.full((4, 2), state['value'])

        class Analyzer(BaseAnalyzer):
            def __init__(self, tag, recalc=False):
                super().__init__(tag, recalc)
                self.derived = Derived()

        a = Analyzer('20240201_000002')
        set_tmpdir(a, self.tmpdir)
        self.assertEqual(a.derived.get_data()[0, 0], 1.0)
        state['value'] = 2.0
        a.recalc()
        self.assertEqual(a.derived.get_data()[0, 0], 2.0)

    @unittest.expectedFailure
    def test_A4_plot_spectra_does_not_modify_cached_psd(self):
        ts = BaseTimeSeries(np.random.default_rng(1).normal(size=(256, 2)),
                            time_vector=np.arange(256) * 1e-3)
        psd_before = ts.power().copy()
        ts.plot_spectra(lineary=True)
        np.testing.assert_allclose(ts.power(), psd_before)

    @unittest.expectedFailure
    def test_A5_missing_optional_time_vector_keeps_data(self):
        fname = os.path.join(self.tmpdir, 'data.npz')
        np.savez(fname, data=np.ones((10, 2)))
        ts = BaseTimeSeries(NumpyDataLoader(fname, key='data'),
                            time_vector=NumpyDataLoader(fname, key='missing_time'))
        self.assertEqual(ts.get_data().shape, (10, 2))


class CrashesTest(_TmpDirTestCase):
    '''B: crashes and API bugs'''

    @unittest.expectedFailure
    def test_B1_arithmetic_on_subclass_with_custom_constructor(self):
        class Slopes(BaseTimeSeries):
            def __init__(self, filename, subap_map):
                super().__init__(np.ones((10, 4)))
                self._subap_map = subap_map
        s = Slopes('unused', None)
        np.testing.assert_array_equal((s + s).get_data(), np.full((10, 4), 2.0))

    @unittest.expectedFailure
    def test_B2_reflected_multiplication(self):
        ts = BaseTimeSeries(np.ones((10, 3)))
        np.testing.assert_array_equal((2 * ts).get_data(), np.full((10, 3), 2.0))

    @unittest.expectedFailure
    def test_B2_reflected_subtraction(self):
        ts = BaseTimeSeries(np.ones((10, 3)))
        np.testing.assert_array_equal((1 - ts).get_data(), np.zeros((10, 3)))

    @unittest.expectedFailure
    def test_B3_power_with_only_from_freq(self):
        ts = BaseTimeSeries(np.random.default_rng(2).normal(size=(100, 2)),
                            time_vector=np.arange(100) * 0.01)
        _ = ts.power(from_freq=1)
        self.assertGreaterEqual(ts.last_cut_frequency().min(), 1 - 1e-9)

    @unittest.expectedFailure
    def test_B4_plot_cumulative_spectra_with_unit_and_no_label(self):
        ts = BaseTimeSeries(np.random.default_rng(3).normal(size=(256, 2)), astropy_unit=u.nm)
        ts.get_data()
        ts.plot_cumulative_spectra()

    @unittest.expectedFailure
    def test_B5_modalplot_to_axes(self):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        modalplot(np.ones(5), np.ones(5), plot_to=ax, title='x')

    @unittest.expectedFailure
    def test_B6_timeseries_from_base_data(self):
        ts = BaseTimeSeries(BaseData(np.ones((10, 3))))
        self.assertEqual(ts.get_data().shape, (10, 3))

    @unittest.expectedFailure
    def test_B7_wiki_with_non_string_values(self):
        class Analyzer(BaseAnalyzer):
            def _info(self):
                return {'snapshot_tag': self._snapshot_tag, 'nframes': 1000}
        Analyzer('20240201_000003').wiki()

    @unittest.expectedFailure
    def test_B8_analyzer_set_append_then_get(self):
        s = BaseAnalyzerSet(['20240201_000004'], file_walker=None, analyzer_type=BaseAnalyzer)
        s.append('20240201_000005')
        self.assertEqual(s.get('20240201_000005').snapshot_tag(), '20240201_000005')

    @unittest.expectedFailure
    def test_B8_empty_analyzer_set_hasattr(self):
        s = BaseAnalyzerSet([], file_walker=None, analyzer_type=BaseAnalyzer)
        self.assertFalse(hasattr(s, 'foo'))

    @unittest.expectedFailure
    def test_B8_analyzer_set_copy(self):
        s = BaseAnalyzerSet(['20240201_000006'], file_walker=None, analyzer_type=BaseAnalyzer)
        self.assertEqual(copy.copy(s).tag_list, s.tag_list)

    @unittest.expectedFailure
    def test_B9_tag_with_suffix(self):
        self.assertEqual(Tag('20240101_120000_KAPA').get_day_as_string(), '20240101')

    @unittest.expectedFailure
    def test_B10_base_indexer_single_positional(self):
        # The docstring promises process_args(5) -> 5
        self.assertEqual(BaseIndexer().process_args(5), 5)


class PerformanceTest(_TmpDirTestCase):
    '''C: performance and memory'''

    @unittest.expectedFailure
    def test_C1_instances_are_garbage_collected(self):
        ts = BaseTimeSeries(np.ones((1000, 100)))
        ts.get_data()
        ref = weakref.ref(ts)
        del ts
        gc.collect()
        self.assertIsNone(ref())

    @unittest.expectedFailure
    def test_C2_members_of_the_same_class_are_all_cached(self):
        class Derived(BaseTimeSeries):
            def __init__(self):
                super().__init__(OnTheFlyLoader(self._calc))

            @cache_on_disk
            def _calc(self):
                return np.ones((10, 2))

        class Analyzer(BaseAnalyzer):
            def __init__(self, tag, recalc=False):
                super().__init__(tag, recalc)
                self.first = Derived()
                self.second = Derived()

        a = Analyzer('20240201_000007')
        self.assertIsNotNone(get_disk_cacher(a.first, Derived._calc)._tag)
        self.assertIsNotNone(get_disk_cacher(a.second, Derived._calc)._tag)

    @unittest.expectedFailure
    def test_C3_base_data_loads_once(self):
        fname = os.path.join(self.tmpdir, 'm.npy')
        np.save(fname, np.eye(3))
        calls = []

        class CountingLoader(NumpyDataLoader):
            def load(self, **kwargs):
                calls.append(1)
                return super().load(**kwargs)

        data = BaseData(CountingLoader(fname))
        for _ in range(5):
            data.get_data()
        _ = data.shape
        self.assertEqual(len(calls), 1)


class CacheRobustnessTest(_TmpDirTestCase):
    '''D: cache safety and robustness'''

    @unittest.expectedFailure
    def test_D1_default_cache_dir_is_not_the_shared_temp_dir(self):
        cacher = DiskCacher(lambda self: 1)
        self.assertNotEqual(cacher._tmpdir, tempfile.gettempdir())

    @unittest.expectedFailure
    def test_D2_truncated_cache_file_is_recomputed(self):
        class Analyzer(BaseAnalyzer):
            @cache_on_disk
            def f(self):
                return {'a': 1}

        a = Analyzer('20240201_000008')
        set_tmpdir(a, self.tmpdir)
        a.f()
        path = get_disk_cacher(a, Analyzer.f).fullpath()
        with open(path, 'wb') as f:
            f.write(pickle.dumps({'a': 1})[:5])   # Simulate an interrupted write

        b = Analyzer('20240201_000008')
        set_tmpdir(b, self.tmpdir)
        self.assertEqual(b.f(), {'a': 1})


if __name__ == "__main__":
    unittest.main()
