#!/usr/bin/env python
import unittest
import numpy as np
from arte.types.scalar_bidimensional_function import \
    ScalarBidimensionalFunction
from arte.types.domainxy import DomainXY


class ScalarBidimensionalFunctionTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCreation(self):
        x = np.tile(np.linspace(-1, 1, 11), (11, 1))
        y = x.T
        v = 42 * (x ** 2 + y ** 2)
        xyFunc = ScalarBidimensionalFunction(v, x, y)
        self.assertEqual(-1, xyFunc.domain.xcoord.min())
        self.assertAlmostEqual(0.2, xyFunc.domain.step[0])
        self.assertAlmostEqual(0.2, xyFunc.domain.step[1])

    def testCutOutFromROI(self):
        x = np.tile(np.linspace(-1, 1, 11), (11, 1))
        v = 42 * x
        xyFunc = ScalarBidimensionalFunction(
            v, domain=DomainXY.from_linspace(-1, 1, 11))

        xmin = 0
        xmax = 0.5
        ymin = -0.5
        ymax = 0.5
        cutOut = xyFunc.get_roi(xmin, xmax, ymin, ymax)
        want = [xmin, xmax, ymin, ymax]
        got = cutOut.extent
        self.assertLessEqual(got[0], want[0])
        self.assertLessEqual(got[2], want[2])
        self.assertGreaterEqual(got[1], want[1])
        self.assertGreaterEqual(got[3], want[3])

    def testCutOutFromSameROI(self):
        x = np.tile(np.linspace(-1, 1, 11), (11, 1))
        v = 42 * x
        xyFunc = ScalarBidimensionalFunction(
            v, domain=DomainXY.from_linspace(-1, 1, 11))

        xmin = -1
        xmax = 1
        ymin = -1
        ymax = 1
        cutOut = xyFunc.get_roi(xmin, xmax, ymin, ymax)
        want = [xmin, xmax, ymin, ymax]
        got = cutOut.extent
        self.assertEqual(want, got)

    def testCutOutFromTooSmallROI(self):
        x = np.tile(np.linspace(-1, 1, 11), (11, 1))
        v = 42 * x
        xyFunc = ScalarBidimensionalFunction(
            v, domain=DomainXY.from_linspace(-1, 1, 11))

        xmin = 0.001
        xmax = 0.002
        ymin = 0.000
        ymax = 0.001
        cutOut = xyFunc.get_roi(xmin, xmax, ymin, ymax)
        self.assertEqual((2, 2), cutOut.shape)

    def testInterpolate(self):
        x = np.tile(np.linspace(-1, 1, 11), (11, 1))
        v = 42 * x
        xyFunc = ScalarBidimensionalFunction(
            v, domain=DomainXY.from_linspace(-1, 1, 11))

        want = np.array([0, 4.2, 8.4])
        got = xyFunc.interpolate_in_xy([0, 0.1, 0.2], [0, 1, 2])
        self.assertTrue(np.allclose(want, got),
                        "%s %s" % (want, got))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
