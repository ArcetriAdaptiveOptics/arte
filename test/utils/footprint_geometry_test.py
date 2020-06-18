#!/usr/bin/env python
import unittest
from arte.utils.footprint_geometry import FootprintGeometry


class FootprintGeometryTest(unittest.TestCase):

    def test_ngs_on_axis_at_ground(self):
        fp = FootprintGeometry()
        fp.set_zenith_angle(0)
        fp.setTelescopeRadiusInMeter(20)
        fp.setLayerAltitude(0)
        fp.setInstrumentFoV(0)
        fp.addTarget(0, 0)
        fp.addNgs(0, 0)
        fp.compute()
        ngs_fp = fp.getNgsFootprint()[0]
        meta_fp = fp.getMetapupilFootprint()[0]
        self.assertAlmostEqual(0, ngs_fp.x)
        self.assertAlmostEqual(0, ngs_fp.y)
        self.assertAlmostEqual(20, ngs_fp.r)
        self.assertAlmostEqual(20, meta_fp.r)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
