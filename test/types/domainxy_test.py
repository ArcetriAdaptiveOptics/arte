# -*- coding: utf-8 -*-

#!/usr/bin/env python
import unittest
import numpy as np
import astropy.units as u
from arte.math.make_xy import make_xy
from arte.types.domainxy import DomainXY

__version__ = "$Id:$"

class DomainXYTest(unittest.TestCase):
    
    def test_extent(self):
        xmin= -1
        xmax= 1.5
        ymin= 2
        ymax= 3
        domain= DomainXY.from_extent(xmin, xmax, ymin, ymax, 11)
        want= [xmin, xmax, ymin, ymax]
        got= domain.extent
        self.assertEqual(want, got)
        

    def test_step(self):
        domain = DomainXY.from_linspace(-1, 1, 11)
        self.assertAlmostEqual(0.2, domain.step[0])
        self.assertAlmostEqual(0.2, domain.step[1])

    def test_crop(self): 
    
        domain = DomainXY.from_linspace(-1, 1, 11)
        xmin= 0
        xmax= 0.5
        ymin= -0.5
        ymax= 0.5
        cropped= domain.cropped(xmin, xmax, ymin, ymax)
        want= [xmin, xmax, ymin, ymax]
        got= cropped.extent
        self.assertLessEqual(got[0], want[0])
        self.assertLessEqual(got[2], want[2])
        self.assertGreaterEqual(got[1], want[1])
        self.assertGreaterEqual(got[3], want[3])

    def test_crop_all(self):
        
        domain = DomainXY.from_linspace(-1, 1, 11)

        xmin= -1
        xmax= 1
        ymin= -1
        ymax= 1
        cropped= domain.cropped(xmin, xmax, ymin, ymax)
        want= [xmin, xmax, ymin, ymax]
        got= cropped.extent
        self.assertEqual(want, got)

    def test_crop_small(self):
        domain = DomainXY.from_linspace(-1, 1, 11)

        xmin= 0.001
        xmax= 0.002
        ymin= 0.000
        ymax= 0.001
        cropped= domain.cropped(xmin, xmax, ymin, ymax)
        self.assertEqual((2,2), cropped.shape)

    def test_makexy(self):
        
        domain = DomainXY.from_makexy(5,2)
        x,y = make_xy(5,2)
        
        np.testing.assert_array_equal(x, domain.xmap)
        np.testing.assert_array_equal(y, domain.ymap)
        
    def test_map(self):

        x = np.arange(2)
        y = np.arange(3)
        domain = DomainXY.from_xy_vectors(x,y)
        
        xmap = np.tile(x, (3,1))
        ymap = np.tile(y, (2,1)).T
        
        np.testing.assert_array_equal(xmap, domain.xmap)
        np.testing.assert_array_equal(ymap, domain.ymap)
        
    def test_getitem(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        domain2 = domain[3:5,2:5]
        
        np.testing.assert_array_equal(domain.xcoord[2:5], domain2.xcoord)
        np.testing.assert_array_equal(domain.ycoord[3:5], domain2.ycoord)

    def test_getitem_singleslice(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        domain2 = domain[2:5]
        
        np.testing.assert_array_equal(domain.ycoord[2:5], domain2.ycoord)
        np.testing.assert_array_equal(domain.xcoord, domain2.xcoord)

    def test_get_crop_slice(self):

        domain = DomainXY.from_linspace(-1, 1, 11)

        sl = domain.get_crop_slice(2,5,3,5)
        
        domain2 = domain[sl]
        domain3 = domain.cropped(2,5,2,5)
        np.testing.assert_array_equal(domain3.xcoord, domain2.xcoord)
        np.testing.assert_array_equal(domain3.ycoord, domain2.ycoord)

    def test_from_shape(self):
        
        domain = DomainXY.from_shape((5,10), pixel_size = 1*u.cm)
        
        assert domain.shape == (5,10)

    def test_units(self):
        
        domain = DomainXY.from_shape((10,10), pixel_size = 1*u.cm)

        assert domain.xcoord.unit == u.cm
        assert domain.xmap.unit == u.cm
        assert domain.origin[0].unit == u.cm
        assert domain.step[0].unit == u.cm
        assert domain[2:4,2:4].xcoord.unit == u.cm
        assert domain.cropped(1,4,1,4).xcoord.unit == u.cm

    def test_cropping_with_units(self):
         
         domain = DomainXY.from_shape((100,100), pixel_size = 1*u.cm)
         
         xmin = 0.1 * u.m
         xmax = 0.2 * u.m
         ymin = 0.1 * u.m
         ymax = 0.2 * u.m

         # 12 = 10 cm + two 0.5cm edges because the linspace
         # is -49.5, -48.5, -47.5, etc.
         assert domain.cropped(xmin, xmax, ymin, ymax).shape == (12,12)


    def test_shift(self):
         
         domain = DomainXY.from_linspace(-1, 1, 11)
         domain.shift(0.2, -0.2)
         assert np.allclose(domain.origin, (4,6))

    def test_shift_w_units(self):
         
         domain = DomainXY.from_shape((100,100), pixel_size = 1*u.cm)
         self.assertAlmostEqual(domain.origin[0], 49.5*u.cm)
         self.assertAlmostEqual(domain.origin[1], 49.5*u.cm)

         domain.shift(0.1*u.m, -3.1415*u.mm)
         self.assertAlmostEqual(domain.origin[0], 39.5*u.cm)
         self.assertAlmostEqual(domain.origin[1], 49.81415*u.cm)
         
         with self.assertRaises(Exception):
             domain.shift(0.1*u.s, 0)
             
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
        domain2 = DomainXY.from_linspace(-1*u.m, 1*u.m, 11)
        
        assert domain1.unit == (1,1)
        assert domain2.unit == (u.m, u.m)

    def test_equal(self):
        
        domain = DomainXY.from_linspace(-1, 1, 11)
        assert domain == domain
        assert not (domain != domain)

        domain2 = domain.shifted(0.1, 0.1)
        assert domain2 != domain
        assert not (domain2 == domain)

    def test_equal_w_units(self):
        
        domain = DomainXY.from_linspace(-1*u.m, 1*u.m, 11)
        assert domain == domain
        assert not (domain != domain)

        domain2 = domain.shifted(0.1, 0.1)
        assert domain2 != domain
        assert not (domain2 == domain)


if __name__ == "__main__":
    unittest.main()
