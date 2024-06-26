# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np
import astropy.units as u

from arte.math.make_xy import make_xy
from arte.types.domainxy import DomainXY


class DomainXYTest(unittest.TestCase):

    def test_extent(self):
        xmin = -1
        xmax = 1.5
        ymin = 2
        ymax = 3
        domain = DomainXY.from_extent(xmin, xmax, ymin, ymax, 11)
        want = [xmin, xmax, ymin, ymax]
        got = domain.extent
        self.assertEqual(want, got)

    def test_step(self):
        domain = DomainXY.from_linspace(-1, 1, 11)
        self.assertAlmostEqual(0.2, domain.step[0])
        self.assertAlmostEqual(0.2, domain.step[1])

    def test_crop(self):

        domain = DomainXY.from_linspace(-1, 1, 11)
        xmin = 0
        xmax = 0.5
        ymin = -0.5
        ymax = 0.5
        cropped = domain.cropped(xmin, xmax, ymin, ymax)
        want = [xmin, xmax, ymin, ymax]
        got = cropped.extent
        self.assertLessEqual(got[0], want[0])
        self.assertLessEqual(got[2], want[2])
        self.assertGreaterEqual(got[1], want[1])
        self.assertGreaterEqual(got[3], want[3])

    def test_crop_all(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        xmin = -1
        xmax = 1
        ymin = -1
        ymax = 1
        cropped = domain.cropped(xmin, xmax, ymin, ymax)
        want = [xmin, xmax, ymin, ymax]
        got = cropped.extent
        self.assertEqual(want, got)

    def test_crop_small(self):
        domain = DomainXY.from_linspace(-1, 1, 11)

        xmin = 0.001
        xmax = 0.002
        ymin = 0.000
        ymax = 0.001
        cropped = domain.cropped(xmin, xmax, ymin, ymax)
        self.assertEqual((2, 2), cropped.shape)

    @unittest.skip("Cropping on exact values depends on floating precision")
    def test_crop_on_values(self):
        domain = DomainXY.from_linspace(-1, 1, 11)

        xmin = 0.6
        xmax = 0.8
        ymin = -0.6
        ymax = -0.2
        cropped = domain.cropped(xmin, xmax, ymin, ymax)
        self.assertEqual((2, 3), cropped.shape)

    def test_crop_w_units(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        with self.assertRaises(AssertionError):
            _ = domain.cropped(*([1, 2, 3, 4] * u.cm))

    def test_crop_missing_units(self):

        domain = DomainXY.from_linspace(-1 * u.cm, 1 * u.cm, 11)

        with self.assertRaises(AssertionError):
            _ = domain.cropped(1, 2, 3, 4)

        with self.assertRaises(AssertionError):
            _ = domain.cropped(1 * u.cm, 2, 3, 4)

        with self.assertRaises(AssertionError):
            _ = domain.cropped(1 * u.cm, 2 * u.cm, 3, 4)

        with self.assertRaises(AssertionError):
            _ = domain.cropped(1 * u.cm, 2 * u.cm, 3 * u.cm, 4)

    def test_makexy(self):

        domain = DomainXY.from_makexy(5, 2)
        x, y = make_xy(5, 2)

        np.testing.assert_array_equal(x, domain.xmap)
        np.testing.assert_array_equal(y, domain.ymap)

    def test_map(self):

        x = np.arange(2)
        y = np.arange(3)
        domain = DomainXY.from_xy_vectors(x, y)

        xmap = np.tile(x, (3, 1))
        ymap = np.tile(y, (2, 1)).T

        np.testing.assert_array_equal(xmap, domain.xmap)
        np.testing.assert_array_equal(ymap, domain.ymap)

    def test_getitem(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        domain2 = domain[3:5, 2:5]

        np.testing.assert_array_equal(domain.xcoord[2:5], domain2.xcoord)
        np.testing.assert_array_equal(domain.ycoord[3:5], domain2.ycoord)

    def test_getitem_singleslice(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        domain2 = domain[2:5]

        np.testing.assert_array_equal(domain.ycoord[2:5], domain2.ycoord)
        np.testing.assert_array_equal(domain.xcoord, domain2.xcoord)

    def test_get_crop_slice(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        sl = domain.get_crop_slice(2, 5, 3, 5)

        domain2 = domain[sl]
        domain3 = domain.cropped(2, 5, 2, 5)
        np.testing.assert_array_equal(domain3.xcoord, domain2.xcoord)
        np.testing.assert_array_equal(domain3.ycoord, domain2.ycoord)

    def test_from_shape(self):
        domain = DomainXY.from_shape((5, 10), pixel_size=1 * u.cm)

        self.assertEqual((5, 10), domain.shape)
        self.assertEqual(-2 * u.cm, domain.ymap[0, 0])
        self.assertEqual(-4.5 * u.cm, domain.xmap[0, 0])
        self.assertEqual(0 * u.cm, domain.ymap[2, 5])
        self.assertEqual(0.5 * u.cm, domain.xmap[2, 5])
        self.assertEqual(2 * u.cm, domain.ymap[4, 9])
        self.assertEqual(4.5 * u.cm, domain.xmap[4, 9])

    def test_from_linspace(self):
        domain = DomainXY.from_linspace(-10, 10, 21)

        self.assertEqual((21, 21), domain.shape)
        self.assertEqual(-10, domain.ymap[0, 0])
        self.assertEqual(-10, domain.xmap[0, 0])
        self.assertEqual(-8, domain.ymap[2, 5])
        self.assertEqual(-5, domain.xmap[2, 5])
        self.assertEqual(0, domain.ymap[10, 10])

    def test_from_shape_two_pixel_size(self):
        domain = DomainXY.from_shape(
            (5, 10), pixel_size=(1 * u.cm, 2 * u.km))

        self.assertEqual((5, 10), domain.shape)
        self.assertEqual(-2 * u.cm, domain.ymap[0, 0])
        self.assertEqual(-9 * u.km, domain.xmap[0, 0])
        self.assertEqual(0 * u.cm, domain.ymap[2, 5])
        self.assertEqual(1.0 * u.km, domain.xmap[2, 5])
        self.assertEqual(2 * u.cm, domain.ymap[4, 9])
        self.assertEqual(9 * u.km, domain.xmap[4, 9])

    def test_units(self):

        domain = DomainXY.from_shape((10, 10), pixel_size=1 * u.cm)

        assert domain.xcoord.unit == u.cm
        assert domain.xmap.unit == u.cm
        assert isinstance(domain.origin[0], float)  # no unit
        assert domain.step[0].unit == u.cm
        assert domain[2:4, 2:4].xcoord.unit == u.cm
        assert domain.cropped(1 * u.cm, 4 * u.cm, 1 * u.cm,
                              4 * u.cm).xcoord.unit == u.cm

    def test_cropping_with_units(self):

        domain = DomainXY.from_shape((100, 100), pixel_size=1 * u.cm)

        xmin = 0.10 * u.m
        xmax = 0.20 * u.m
        ymin = 0.40 * u.m
        ymax = 0.45 * u.m

        cropped = domain.cropped(xmin, xmax, ymin, ymax)

        # 12 = 10 cm * 1pixel/cm + two 0.5cm edges because the linspace
        # Same for 7 = 5cm * 1pixel/cm + two 0.5cm edges
        assert cropped.shape == (7, 12)
        assert cropped.extent[0] == 9.5 * u.cm
        assert cropped.extent[1] == 20.5 * u.cm
        assert cropped.extent[2] == 39.5 * u.cm
        assert cropped.extent[3] == 45.5 * u.cm

    def test_shift(self):

        domain = DomainXY.from_linspace(-1, 1, 11)
        domain.shift(0.2, -0.2)
        assert np.allclose(domain.origin, (4, 6))

        # Raise if shift with unit
        with self.assertRaises(AssertionError):
            domain.shift(0.1 * u.m, 0.2 * u.m)

    def test_shift_w_units(self):

        domain = DomainXY.from_shape((100, 100), pixel_size=1 * u.cm)
        self.assertAlmostEqual(domain.origin[0], 49.5)
        self.assertAlmostEqual(domain.origin[1], 49.5)

        domain.shift(0.1 * u.m, -3.1415 * u.mm)
        self.assertAlmostEqual(domain.origin[0], 39.5)
        self.assertAlmostEqual(domain.origin[1], 49.81415)

        # Raise if incompatible unit
        with self.assertRaises(AssertionError):
            domain.shift(0.1 * u.s, 0.2 * u.s)

        # Raise if missing unit
        with self.assertRaises(AssertionError):
            domain.shift(0.1 * u.m, 0.2)

        # Raise if missing unit
        with self.assertRaises(AssertionError):
            domain.shift(0.1, 0.2 * u.m)

    def test_copy(self):

        domain = DomainXY.from_linspace(-1, 1, 11)
        domain2 = domain[:]

        np.testing.assert_array_equal(domain.xcoord, domain2.xcoord)
        np.testing.assert_array_equal(domain.ycoord, domain2.ycoord)

    def tet_shifted(self):

        domain = DomainXY.from_linspace(-1, 1, 11)
        domain2 = domain.shifted(0.1, 0.2)
        domain.shift(0.1, 0.2)

        assert domain == domain2

    def test_unit_property(self):

        domain1 = DomainXY.from_linspace(-1, 1, 11)
        domain2 = DomainXY.from_linspace(-1 * u.m, 1 * u.m, 11)

        assert domain1.unit == (1, 1)
        assert domain2.unit == (u.m, u.m)

    def test_equal(self):

        domain = DomainXY.from_linspace(-1, 1, 11)
        assert domain == domain
        assert not (domain != domain)

        domain2 = domain.shifted(0.1, 0.1)
        assert domain2 != domain
        assert not (domain2 == domain)

    def test_equal_w_units(self):

        domain = DomainXY.from_linspace(-1 * u.m, 1 * u.m, 11)
        assert domain == domain
        assert not (domain != domain)

        domain2 = domain.shifted(0.1 * u.cm, 0.1 * u.km)
        assert domain2 != domain
        assert not (domain2 == domain)

    def test_contains(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        self.assertTrue(domain.contains(-0.5, 0))
        self.assertFalse(domain.contains(2, 0))
        self.assertFalse(domain.contains(4, 2))

        # Raise if we get a with unit
        with self.assertRaises(AssertionError):
            domain.shift(0.1 * u.m, 0.2 * u.m)

    def test_contains_w_units(self):

        domain = DomainXY.from_linspace(-1 * u.mm, 1 * u.mm, 11)

        self.assertTrue(domain.contains(-0.5 * u.mm, 0 * u.mm))
        self.assertFalse(domain.contains(2 * u.mm, 0 * u.mm))
        self.assertFalse(domain.contains(4 * u.mm, 2 * u.mm))

        # Raise if incompatible unit
        with self.assertRaises(AssertionError):
            domain.contains(-0.5 * u.s, 0 * u.kg)

        # Raise if missing unit
        with self.assertRaises(AssertionError):
            domain.contains(-0.5, 0 * u.mm)

        # Raise if missing unit
        with self.assertRaises(AssertionError):
            domain.contains(-0.5 * u.mm, 0)

    def test_boundingbox(self):

        domain = DomainXY.from_linspace(-1, 1, 11)
        box = domain.boundingbox(0.32, -0.25, span=1)

        np.testing.assert_array_almost_equal(box.xcoord, [0.2, 0.4])
        np.testing.assert_array_almost_equal(box.ycoord, [-0.4, -0.2])

        # Raise if we get a with unit
        with self.assertRaises(AssertionError):
            domain.boundingbox(0.1 * u.m, 0.2 * u.m, span=1)

    def test_boundingbox_w_units(self):

        domain = DomainXY.from_linspace(-1 * u.mm, 1 * u.mm, 11)
        box = domain.boundingbox(0.32 * u.mm, -0.25 * u.mm, span=1)

        box_xcoord = box.xcoord.to(u.mm).data
        box_ycoord = box.ycoord.to(u.mm).data
        np.testing.assert_array_almost_equal(box_xcoord, [0.2, 0.4])
        np.testing.assert_array_almost_equal(box_ycoord, [-0.4, -0.2])

        # Raise if incompatible unit
        with self.assertRaises(AssertionError):
            domain.boundingbox(-0.5 * u.s, 0 * u.kg)

        # Raise if missing unit
        with self.assertRaises(AssertionError):
            domain.boundingbox(-0.5, 0 * u.mm)

        # Raise if missing unit
        with self.assertRaises(AssertionError):
            domain.boundingbox(-0.5 * u.mm, 0)


if __name__ == "__main__":
    unittest.main()
