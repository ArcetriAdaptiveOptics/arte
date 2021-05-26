import unittest
from arte.photometry.filters import Filters
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose


class FiltersTest(unittest.TestCase):


    def testEsoEtc(self):
        assert_quantity_allclose(
            Filters.get(Filters.ESO_ETC_U).avgwave(),
            3582*u.angstrom,
            atol=1*u.angstrom)

        assert_quantity_allclose(
            Filters.get(Filters.ESO_ETC_V).avgwave(),
            5492*u.angstrom,
            atol=1*u.angstrom)

        assert_quantity_allclose(
            Filters.get(Filters.ESO_ETC_Z).equivwidth(),
            1063*u.angstrom,
            atol=1*u.angstrom)

        assert_quantity_allclose(
            Filters.get(Filters.ESO_ETC_I).efficiency(),
            0.1764,
            atol=0.001)

    def testSynphot(self):
        assert_quantity_allclose(
            Filters.get(Filters.JOHNSON_I).efficiency(),
            0.266,
            atol=0.001)


if __name__ == "__main__":
    unittest.main()
