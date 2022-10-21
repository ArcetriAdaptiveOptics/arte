#!/usr/bin/env python
import unittest
import numpy as np
from arte.types.mask import CircularMask, AnnularMask




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


    def testCreateCircularMaskFromMaskedArrayWithAReallyCircularMask(self):
        shape= (140, 100)
        aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 40))
        maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
        retrievedMask= CircularMask.fromMaskedArray(maskedArray)
        np.testing.assert_allclose(
            aMask.radius(), retrievedMask.radius(), rtol=0.01)
        np.testing.assert_allclose(
            aMask.center(), retrievedMask.center(), atol=1)
        self.assertTrue(np.in1d(retrievedMask.in_mask_indices(),
                                aMask.in_mask_indices()).all())


    def testCreateCircularMaskFromMaskedArrayWithFloats(self):
        aMask = CircularMask((486, 640), 126.32, (235.419, 309.468))
        marray = np.ma.array(data = np.zeros((486, 640)), mask=aMask.mask())
        
        retrievedMask = CircularMask.fromMaskedArray(marray)
        
        np.testing.assert_allclose(
            aMask.radius(), retrievedMask.radius(), rtol=0.01)
        np.testing.assert_allclose(
            aMask.center(), retrievedMask.center(), atol=1)        
        self.assertTrue(np.in1d(retrievedMask.in_mask_indices(),
                                aMask.in_mask_indices()).all())


    def testCreateCircularMaskFromMaskedArrayWithNonCircularMask(self):
        shape= (140, 100)
        aSquareMask= np.ones((shape))
        aSquareMask[40:80, 20:60]= 0
        maskedArray= np.ma.masked_array(np.ones(shape),
                                        mask=aSquareMask.astype(bool))
        retrievedMask= CircularMask.fromMaskedArray(maskedArray)
        self.assertTrue(
            np.in1d(retrievedMask.in_mask_indices(),
                    np.argwhere(aSquareMask.flatten() == False)).all())


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
        
    def testAnularMask(self):
        mask1= CircularMask((10, 10),4,(5,5))
        mask2= CircularMask((10, 10),2,(5,5))
        mask3 = AnnularMask ((10, 10),4,(5,5),2)
        mask4 = mask1.mask() | ~mask2.mask()
        self.assertEqual(mask3.mask().all(), mask4.all())
        
        



if __name__ == "__main__":
    unittest.main()
