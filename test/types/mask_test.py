#!/usr/bin/env python
import unittest
import numpy as np
from arte.types.mask import CircularMask


__version__ = "$Id: $"


class MaskTest(unittest.TestCase):


    def testStandard(self):
        mask= CircularMask((13, 14))
        self.assertEqual(mask.shape(), (13, 14))


    def testNullRadiusMakeNullMask(self):
        mask= CircularMask((10, 20), maskRadius=0)
        self.assertTrue(np.all(mask.mask()))


    def testPassingRadiusNone(self):
        mask= CircularMask((10, 10))
        self.assertEqual(5, mask.radius())


    def testPassingCenterNone(self):
        mask= CircularMask((20, 10))
        self.assertTrue(np.allclose([10, 5],
                                    mask.center()),
                        "center is %s" % mask.center())


    def testCreateCircularMaskFromMaskedArray(self):
        shape= (140, 100)
        aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 40))
        maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
        retrievedMask= CircularMask.fromMaskedArray(maskedArray)
        self.assertEqual(aMask.radius(), retrievedMask.radius())
        self.assertTrue(np.array_equal(aMask.center(), retrievedMask.center()))


    def testCreateCircularMaskFromMaskedArrayRaisesIfMaskIsNotCircular(self):
        shape= (140, 100)
        aSquareMask= np.ones((shape))
        aSquareMask[40:80, 20:60]= 0
        maskedArray= np.ma.masked_array(np.ones(shape), mask=aSquareMask)
        self.assertRaises(Exception,
                          CircularMask.fromMaskedArray, maskedArray)


    def testRegionOfInterest(self):
        shape= (140, 100)
        aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 45))
        roi= aMask.regionOfInterest()
        self.assertEqual(40, roi.ymin)
        self.assertEqual(80, roi.ymax)
        self.assertEqual(25, roi.xmin)
        self.assertEqual(65, roi.xmax)


    def testAsTransmissionValue(self):
        mask= CircularMask((10, 10))
        transmission= mask.asTransmissionValue()
        self.assertEqual(1, transmission[5, 5])
        self.assertEqual(0, transmission[0, 0])

if __name__ == "__main__":
    unittest.main()
