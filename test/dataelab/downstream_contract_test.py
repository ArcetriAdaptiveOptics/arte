# -*- coding: utf-8 -*-
'''
Contract tests for the arte.dataelab / arte.time_series API used downstream.

These tests pin the behaviour that downstream projects (KADAT, ekarus, prima,
membranemirror, MORFEO, M4) rely on today. They must keep passing, unchanged,
when dataelab is extracted into a standalone package and arte re-exports it
(see issue #78, phases 1-3). They intentionally exercise the same patterns
found in those projects: subclassing, extension points, lazy loaders,
disk caching, NotAvailable propagation and a few private attributes that
downstream code touches directly.

Tests documenting known defects live in known_defects_test.py.
'''

import os
import shutil
import tempfile
import unittest
import importlib

import numpy as np
import astropy.units as u

import matplotlib
matplotlib.use('Agg')   # Draw in background

from arte.dataelab.base_timeseries import BaseTimeSeries
from arte.dataelab.base_data import BaseData
from arte.dataelab.base_2dmap import Base2dMap
from arte.dataelab.base_slopes import BaseSlopes
from arte.dataelab.base_analyzer import BaseAnalyzer
from arte.dataelab.base_analyzer_set import BaseAnalyzerSet
from arte.dataelab.base_file_walker import AbstractFileNameWalker
from arte.dataelab.cache_on_disk import cache_on_disk, set_tmpdir, get_disk_cacher
from arte.dataelab.data_loader import NumpyDataLoader, FitsDataLoader, OnTheFlyLoader
from arte.dataelab.tag import Tag
from arte.utils.not_available import NotAvailable


# Every (module, name) pair imported by downstream projects.
DOWNSTREAM_IMPORTS = [
    ('arte.dataelab.analyzer_plots', 'modalplot'),
    ('arte.dataelab.base_2dmap', 'Base2dMap'),
    ('arte.dataelab.base_analyzer', 'BaseAnalyzer'),
    ('arte.dataelab.base_analyzer_set', 'BaseAnalyzerSet'),
    ('arte.dataelab.base_data', 'BaseData'),
    ('arte.dataelab.base_file_walker', 'AbstractFileNameWalker'),
    ('arte.dataelab.base_indexer', 'BaseIndexer'),
    ('arte.dataelab.base_slopes', 'BaseSlopes'),
    ('arte.dataelab.base_timeseries', 'BaseTimeSeries'),
    ('arte.dataelab.cache_on_disk', 'cache_on_disk'),
    ('arte.dataelab.data_loader', 'FitsDataLoader'),
    ('arte.dataelab.data_loader', 'NumpyDataLoader'),
    ('arte.dataelab.data_loader', 'OnTheFlyLoader'),
    ('arte.dataelab.tag', 'Tag'),
    ('arte.time_series.indexer', 'ModeIndexer'),
    ('arte.time_series.indexer', 'RowColIndexer'),
    ('arte.time_series.time_series', 'TimeSeries'),
    ('arte.utils.not_available', 'NotAvailable'),
]


class ImportSurfaceTest(unittest.TestCase):

    def test_downstream_imports_exist(self):
        for module_name, name in DOWNSTREAM_IMPORTS:
            with self.subTest(module=module_name, name=name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, name))

    def test_private_cache_names_are_reachable(self):
        # Needed by the arte re-export layer: it must preserve these names
        module = importlib.import_module('arte.dataelab.cache_on_disk')
        for name in ['DiskCacher', '_discover_cachers', 'set_tag', 'clear_cache',
                     'set_tmpdir', 'set_prefix', 'get_disk_cacher']:
            with self.subTest(name=name):
                self.assertTrue(hasattr(module, name))


class _TmpDirTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _npz(self, name='snapshot.npz', **arrays):
        fname = os.path.join(self.tmpdir, name)
        np.savez(fname, **arrays)
        return fname


class LoaderContractTest(_TmpDirTestCase):

    def test_npz_key_and_postprocess_kwarg(self):
        fname = self._npz(slopes=np.arange(12).reshape(6, 2))
        loader = NumpyDataLoader(fname, key='slopes', postprocess=lambda d: d * 2)
        loader.assert_exists()
        np.testing.assert_array_equal(loader.load(), np.arange(12).reshape(6, 2) * 2)

    def test_postprocess_assigned_after_construction(self):
        # ekarus and prima assign the private attribute directly
        fname = self._npz(modes=np.ones((5, 3, 1)))
        loader = NumpyDataLoader(fname, key='modes')
        loader._postprocess = np.squeeze
        self.assertEqual(loader.load().shape, (5, 3))

    def test_private_attributes_used_by_subclasses(self):
        # ekarus' NumpyDataLoader subclass reads these attributes
        fname = self._npz(a=np.ones(3))
        loader = NumpyDataLoader(fname, key='a', transpose_axes=None)
        self.assertEqual(loader._filename, fname)
        self.assertEqual(loader._key, 'a')
        self.assertIsNone(loader._transpose_axes)

    def test_subclass_overriding_load_without_arguments(self):
        # KADAT timestamp loader pattern
        fname = self._npz(ts=np.arange(5) * 1e9)

        class TimestampLoader(NumpyDataLoader):
            def load(self):
                return super().load() / 1e9 * u.s

        ts = BaseTimeSeries(np.zeros((5, 2)), time_vector=TimestampLoader(fname, key='ts'))
        np.testing.assert_allclose(ts.get_time_vector().to_value(u.s), np.arange(5))
        self.assertEqual(ts.delta_time, 1 * u.s)

    def test_npz_default_key(self):
        fname = os.path.join(self.tmpdir, 'x.npz')
        np.savez(fname, np.arange(4))
        np.testing.assert_array_equal(NumpyDataLoader(fname).load(), np.arange(4))

    def test_fits_loader(self):
        from astropy.io import fits
        fname = os.path.join(self.tmpdir, 'x.fits')
        fits.writeto(fname, np.arange(6.0).reshape(3, 2))
        np.testing.assert_array_equal(FitsDataLoader(fname).load(), np.arange(6.0).reshape(3, 2))

    def test_filename_or_path_accepted(self):
        from pathlib import Path
        fname = os.path.join(self.tmpdir, 'x.npy')
        np.save(fname, np.ones((4, 2)))
        for f in (fname, Path(fname)):
            with self.subTest(type=type(f)):
                self.assertEqual(BaseTimeSeries(f).get_data().shape, (4, 2))
                self.assertEqual(BaseData(f).get_data().shape, (4, 2))


class TimeSeriesSubclassContractTest(_TmpDirTestCase):
    '''Extension points overridden by downstream BaseTimeSeries subclasses'''

    def _slopes_class(self):
        class Slopes(BaseSlopes):
            def __init__(self, filename, subap_geometry):
                super().__init__(NumpyDataLoader(filename, key='slopes'),
                                 astropy_unit=u.arcsec,
                                 axes=('time', 'slopes'))
                self._subap_geometry = subap_geometry

            def get_index_of(self, *args, **kwargs):
                return self._indexer.interleaved_xy(*args, **kwargs)

            def _get_display_cube(self, data_to_display):
                sx = self._subap_geometry.remap_image(data_to_display[:, 0::2])
                sy = self._subap_geometry.remap_image(data_to_display[:, 1::2])
                return np.hstack((sx, sy)).transpose(0, 2, 1)

            def get_display_axes(self):
                return ('time', 'subap', 'subap')
        return Slopes

    def test_subclass_with_geometry(self):
        subap_map = Base2dMap(np.array([[0, 1], [1, 1]]))
        nsubap = subap_map.nvalid
        data = np.arange(10 * 2 * nsubap, dtype=float).reshape(10, 2 * nsubap)
        fname = self._npz(slopes=data)
        slopes = self._slopes_class()(fname, subap_map)

        self.assertEqual(slopes.get_data().unit, u.arcsec)
        np.testing.assert_array_equal(slopes.get_data('x').value, data[:, 0::2])
        np.testing.assert_array_equal(slopes.get_data('y').value, data[:, 1::2])
        self.assertEqual(slopes.get_display().shape, (10, 2, 4))
        self.assertEqual(slopes.get_display_axes(), ('time', 'subap', 'subap'))
        _ = slopes.imshow()

    def test_derived_series_from_bound_method(self):
        # KADAT ResidualDMCommand pattern: data=self._calc, BaseData with axes
        class Derived(BaseTimeSeries):
            def __init__(self, series, rec):
                self._series = series
                self._rec = rec
                super().__init__(data=self._calc, axes=('time', 'actuators'))

            def _calc(self):
                slopes = self._series.get_data(axes=('time', 'slopes'))
                rec = self._rec.get_data(axes=('slopes', 'actuators'))
                return slopes @ rec

        series = BaseTimeSeries(np.ones((7, 4)), axes=('time', 'slopes'))
        rec = BaseData(np.ones((3, 4)), axes=('actuators', 'slopes'))
        derived = Derived(series, rec)
        np.testing.assert_array_equal(derived.get_data(), np.full((7, 3), 4.0))

    def test_derived_series_from_on_the_fly_loader(self):
        # membranemirror pattern
        values = np.random.default_rng(1).normal(size=(20, 3))
        ts = BaseTimeSeries(OnTheFlyLoader(lambda: values),
                            time_vector=OnTheFlyLoader(lambda: np.arange(20) * 0.1),
                            data_label='Wavefront map')
        np.testing.assert_array_equal(ts.get_data(), values)
        self.assertEqual(ts.data_label(), 'Wavefront map')
        self.assertAlmostEqual(ts.delta_time, 0.1)

    def test_statistics_used_downstream(self):
        values = np.random.default_rng(2).normal(size=(200, 4))
        ts = BaseTimeSeries(values * u.nm, time_vector=np.arange(200) * 0.01)
        expected_std = values.std(axis=0)
        np.testing.assert_allclose(ts.get_time_std().to_value(u.nm), expected_std)
        np.testing.assert_allclose(np.asarray(ts.time_std.value).squeeze(), expected_std)
        np.testing.assert_allclose(np.asarray(ts.time_mean.value).squeeze(), values.mean(axis=0))
        self.assertEqual(ts.ensemble_size(), 4)
        self.assertEqual(ts.time_size(), 200)

    def test_times_selection(self):
        ts = BaseTimeSeries(np.arange(10.0)[:, None], time_vector=np.arange(10.0))
        np.testing.assert_array_equal(ts.get_data(times=[2, 5]).ravel(), [2, 3, 4])
        np.testing.assert_array_equal(np.asarray(ts.with_times([2, 5]).value).ravel(), [2, 3, 4])

    def test_power_satisfies_parseval(self):
        # The one-sided Welch PSD integrated over frequency equals the variance
        rng = np.random.default_rng(3)
        n, dt = 4096, 1e-3
        values = rng.normal(scale=2.0, size=(n, 2))
        ts = BaseTimeSeries(values, time_vector=np.arange(n) * dt)
        psd = ts.power()
        freq = ts.frequency()
        df = freq[1] - freq[0]
        np.testing.assert_allclose(psd.sum(axis=0) * df, values.var(axis=0), rtol=1e-2)
        self.assertAlmostEqual(freq[-1], 0.5 / dt)

    def test_power_frequency_cut(self):
        n, dt = 1024, 1e-3
        ts = BaseTimeSeries(np.random.default_rng(4).normal(size=(n, 1)),
                            time_vector=np.arange(n) * dt)
        psd = ts.power(from_freq=10, to_freq=100)
        freq = ts.last_cut_frequency()
        self.assertEqual(len(psd), len(freq))
        self.assertGreaterEqual(freq.min(), 10 - 1e-9)
        self.assertLessEqual(freq.max(), 100 + 1e-9)

    def test_plots_run(self):
        ts = BaseTimeSeries(np.random.default_rng(5).normal(size=(256, 2)),
                            time_vector=np.arange(256) * 1e-3,
                            astropy_unit=u.nm, data_label='modes')
        ts.plot_spectra()
        ts.plot_cumulative_spectra()
        ts.plot_cumulative_spectra(plot_rms=True, from_high_frequency=True)
        _ = ts.tile()

    def test_arithmetic_same_default_constructor(self):
        a = BaseTimeSeries(np.full((5, 2), 3.0))
        b = BaseTimeSeries(np.full((5, 2), 1.0))
        np.testing.assert_array_equal((a - b).get_data(), np.full((5, 2), 2.0))
        np.testing.assert_array_equal((a * 2).get_data(), np.full((5, 2), 6.0))
        np.testing.assert_array_equal((-a).get_data(), np.full((5, 2), -3.0))


class BaseDataContractTest(_TmpDirTestCase):

    def test_callable_data_with_unit_and_axes(self):
        # KADAT SlopesZernikeRec pattern
        class Rec(BaseData):
            def __init__(self):
                super().__init__(self._calc, astropy_unit=u.m / u.arcsec,
                                 axes=('slopes', 'zernikes'))

            def _calc(self):
                return np.arange(6.0).reshape(3, 2)
        rec = Rec()
        self.assertEqual(rec.get_data().unit, u.m / u.arcsec)
        self.assertEqual(rec.get_data(axes=('zernikes', 'slopes')).shape, (2, 3))
        self.assertEqual(rec.shape, (3, 2))

    def test_base2dmap_from_file_for_geometry(self):
        fname = os.path.join(self.tmpdir, 'subapmap.txt')
        np.savetxt(fname, np.array([[0, 1, 0], [1, 1, 1]]))
        geometry = Base2dMap(fname)
        self.assertEqual(geometry.nvalid, 4)
        self.assertEqual(geometry.remap_image(np.arange(4)).shape, (1, 2, 3))


class NotAvailableContractTest(_TmpDirTestCase):

    def test_missing_file_gives_not_available(self):
        missing = os.path.join(self.tmpdir, 'missing.npz')
        for obj in (BaseTimeSeries(missing), BaseData(missing)):
            with self.subTest(type=type(obj)):
                # Downstream checks both isinstance() and str() == 'NA'
                self.assertIsInstance(obj, NotAvailable)
                self.assertEqual(str(obj), 'NA')

    def test_missing_npz_key_gives_not_available(self):
        fname = self._npz(a=np.ones(3))
        self.assertIsInstance(BaseTimeSeries(NumpyDataLoader(fname, key='b')), NotAvailable)

    def test_not_available_class_identity(self):
        # The class produced by dataelab must be the one exported by
        # arte.utils.not_available, otherwise downstream isinstance() checks
        # fail silently after the extraction.
        import arte.dataelab.base_timeseries as bts
        self.assertIs(bts.NotAvailable, NotAvailable)
        self.assertIs(importlib.import_module('arte.dataelab.base_data').NotAvailable, NotAvailable)


class _Walker(AbstractFileNameWalker):

    def __init__(self, root, tags):
        self._root = root
        self._tags = tags

    def snapshot_dir(self, tag):
        return os.path.join(self._root, Tag(tag).get_day_as_string(), str(tag))

    def data_file(self, tag):
        return os.path.join(self.snapshot_dir(tag), 'data.npz')

    def find_tag_between_dates(self, tag_start, tag_stop):
        return [t for t in self._tags if tag_start <= t <= tag_stop]


class AnalyzerContractTest(_TmpDirTestCase):

    TAGS = ['20240101_000001', '20240101_000002', '20240102_000003']

    def setUp(self):
        super().setUp()
        self.walker = _Walker(self.tmpdir, self.TAGS)
        for i, tag in enumerate(self.TAGS[:2]):
            os.makedirs(self.walker.snapshot_dir(tag))
            np.savez(self.walker.data_file(tag), modes=np.full((10, 3), float(i + 1)))
        # The last tag has no data on disk

    def _analyzer_class(self, walker, tmpdir):
        calls = []

        class Modes(BaseTimeSeries):
            def __init__(self, filename):
                super().__init__(NumpyDataLoader(filename, key='modes'),
                                 axes=('time', 'modes'))

        class Squared(BaseTimeSeries):
            def __init__(self, modes):
                self._modes = modes
                super().__init__(data=self._calc, axes=('time', 'modes'))

            @cache_on_disk
            def _calc(self):
                calls.append(1)
                return self._modes.get_data() ** 2

        class Analyzer(BaseAnalyzer):
            def __init__(self, tag, recalc=False, file_walker=None):
                super().__init__(tag, recalc)
                self.modes = Modes(file_walker.data_file(tag))
                self.squared = Squared(self.modes)
                # Must be called after all members exist
                set_tmpdir(self, tmpdir)

            def _info(self):
                info = super()._info()
                info['nframes'] = str(len(self.modes.get_data()))
                return info

        return Analyzer, calls

    def test_get_returns_cached_instance(self):
        Analyzer, _ = self._analyzer_class(self.walker, self.tmpdir)
        a1 = Analyzer.get(self.TAGS[0], file_walker=self.walker)
        a2 = Analyzer.get(self.TAGS[0], file_walker=self.walker)
        self.assertIs(a1, a2)
        self.assertEqual(a1.snapshot_tag(), self.TAGS[0])
        self.assertEqual(a1.info()['nframes'], '10')
        a1.summary()
        a1.wiki()

    def test_disk_cache_persists_across_instances(self):
        cache_dir = os.path.join(self.tmpdir, 'cache')
        Analyzer, calls = self._analyzer_class(self.walker, cache_dir)
        a1 = Analyzer(self.TAGS[0], file_walker=self.walker)
        np.testing.assert_array_equal(a1.squared.get_data(), np.ones((10, 3)))
        self.assertEqual(len(calls), 1)

        # A new instance with the same tag must read the value from disk
        a2 = Analyzer(self.TAGS[0], file_walker=self.walker)
        np.testing.assert_array_equal(a2.squared.get_data(), np.ones((10, 3)))
        self.assertEqual(len(calls), 1)

    def test_cache_file_layout(self):
        # Existing caches stay valid after the extraction only if the file
        # layout (tmpdir/prefix+tag/instance_path.qualname.npy) is unchanged.
        cache_dir = os.path.join(self.tmpdir, 'cache')
        Analyzer, _ = self._analyzer_class(self.walker, cache_dir)
        a = Analyzer(self.TAGS[0], file_walker=self.walker)
        cacher = get_disk_cacher(a.squared, type(a.squared)._calc)
        expected = os.path.join(cache_dir, 'cache' + self.TAGS[0],
                                'root.squared.' + type(a.squared)._calc.__qualname__ + '.npy')
        self.assertEqual(cacher.fullpath(), expected)
        a.squared.get_data()
        self.assertTrue(os.path.exists(expected))

    def test_recalc_recomputes_in_new_instance(self):
        cache_dir = os.path.join(self.tmpdir, 'cache')
        Analyzer, calls = self._analyzer_class(self.walker, cache_dir)
        Analyzer(self.TAGS[0], file_walker=self.walker).squared.get_data()
        Analyzer(self.TAGS[0], recalc=True, file_walker=self.walker).squared.get_data()
        self.assertEqual(len(calls), 2)

    def test_analyzer_set_by_range_and_forwarding(self):
        Analyzer, _ = self._analyzer_class(self.walker, self.tmpdir)
        aset = BaseAnalyzerSet('20240101_000000', '20240101_999999',
                               file_walker=self.walker, analyzer_type=Analyzer)
        aset.set_analyzer_args(file_walker=self.walker)
        self.assertEqual(aset.tag_list, self.TAGS[:2])
        self.assertEqual(len(aset), 2)
        stds = aset.modes.get_time_std()
        self.assertEqual(len(stds), 2)
        self.assertIs(aset[0], aset[self.TAGS[0]])
        self.assertEqual([a.snapshot_tag() for a in aset], self.TAGS[:2])

    def test_analyzer_set_remove_invalids(self):
        # An analyzer whose data is missing propagates NotAvailable
        # to its members; remove_invalids() relies on str() == 'NA'
        class Analyzer(BaseAnalyzer):
            def __init__(self, tag, recalc=False, file_walker=None):
                super().__init__(tag, recalc)
                if not os.path.exists(file_walker.data_file(tag)):
                    NotAvailable.transformInNotAvailable(self)

        aset = BaseAnalyzerSet(self.TAGS, file_walker=self.walker, analyzer_type=Analyzer)
        aset.set_analyzer_args(file_walker=self.walker)
        aset.remove_invalids()
        self.assertEqual(aset.tag_list, self.TAGS[:2])

    def test_apply_w_args(self):
        # KADAT's AnalyzerSet calls this private method
        received = []

        class Analyzer(BaseAnalyzer):
            def record(self, x, label=None):
                received.append((self.snapshot_tag(), x, label))

        aset = BaseAnalyzerSet(self.TAGS[:2], file_walker=self.walker, analyzer_type=Analyzer)
        aset._apply_w_args('record', [[1], [2]], [{'label': 'a'}, {'label': 'b'}])
        self.assertEqual(received, [(self.TAGS[0], 1, 'a'), (self.TAGS[1], 2, 'b')])


class ModalPlotContractTest(unittest.TestCase):

    def test_modalplot_with_units(self):
        from arte.dataelab.analyzer_plots import modalplot
        plt = modalplot(np.arange(1, 6) * u.nm, np.arange(1, 6) * 1e-3 * u.um,
                        title='test')
        self.assertIsNotNone(plt)


if __name__ == "__main__":
    unittest.main()
