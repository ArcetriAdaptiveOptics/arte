#!/usr/bin/env python
# test class for RBFGenerator class deriving from unittest.TestCase
import unittest
import numpy as np
from arte.utils.rbf_generator import RBFGenerator
from arte.types.mask import CircularMask
from numpy.testing import assert_allclose

class TestRBFGenerator(unittest.TestCase):

    def setUp(self):
        self._coords = [(10,10), (20,20), (30,30)]
        self._mask = CircularMask((128,128), 64, (64,64))
        self._rbf = RBFGenerator(self._mask, self._coords, 'TPS_RBF')
        self._rbf1 = RBFGenerator(self._mask, self._coords, 'GAUSS_RBF')
        self._rbf2 = RBFGenerator(self._mask, self._coords, 'INV_QUADRATIC')
        self._rbf3 = RBFGenerator(self._mask, self._coords, 'MULTIQUADRIC')


    def testGenerate(self):
        self._rbf.generate()
        self.assertEqual(self._rbf._rbfBase.shape, (128, 128, 3))
        self._rbf1.generate()
        self.assertEqual(self._rbf1._rbfBase.shape, (128, 128, 3))
        self._rbf2.generate()
        self.assertEqual(self._rbf2._rbfBase.shape, (128, 128, 3))
        self._rbf3.generate()
        self.assertEqual(self._rbf3._rbfBase.shape, (128, 128, 3))


    def testGetRBF(self):
        self._rbf.generate()
        rbf = self._rbf.getRBF(0)
        self.assertEqual(rbf.shape, (128, 128))
        self._rbf1.generate()
        rbf1 = self._rbf1.getRBF(0)
        self.assertEqual(rbf1.shape, (128, 128))
        self._rbf2.generate()
        rbf2 = self._rbf2.getRBF(0)
        self.assertEqual(rbf2.shape, (128, 128))
        self._rbf3.generate()
        rbf3 = self._rbf3.getRBF(0)
        self.assertEqual(rbf3.shape, (128, 128))



    def testGetRBFDict(self):
        self._rbf.generate()
        rbf_dict = self._rbf.getRBFDict([0, 2])
        self.assertEqual(len(rbf_dict), 2)
        self._rbf1.generate()
        rbf_dict1 = self._rbf1.getRBFDict([0, 2])
        self.assertEqual(len(rbf_dict1), 2)
        self._rbf2.generate()
        rbf_dict2 = self._rbf2.getRBFDict([0, 2])
        self.assertEqual(len(rbf_dict2), 2)
        self._rbf3.generate()
        rbf_dict3 = self._rbf3.getRBFDict([0, 2])
        self.assertEqual(len(rbf_dict3), 2)



    def testGetRBFCube(self):
        self._rbf.generate()
        rbf_cube = self._rbf.getRBFCube()
        self.assertEqual(rbf_cube.shape, (128, 128, 3))
        #write test for other rbf functions
        self._rbf1.generate()
        rbf_cube1 = self._rbf1.getRBFCube()
        self.assertEqual(rbf_cube1.shape, (128, 128, 3))
        self._rbf2.generate()
        rbf_cube2 = self._rbf2.getRBFCube()
        self.assertEqual(rbf_cube2.shape, (128, 128, 3))
        self._rbf3.generate()
        rbf_cube3 = self._rbf3.getRBFCube()
        self.assertEqual(rbf_cube3.shape, (128, 128, 3))

    

if __name__ == "__main__":
    unittest.main()





