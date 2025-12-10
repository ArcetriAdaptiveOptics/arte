import unittest
import numpy as np
from arte.photometry.transmissive_elements import (
    TransmissiveElement, TransmissiveSystem, Direction, Bandpass
)
from arte.photometry.transmissive_elements_catalogs import (
    GlassesCatalog, CoatingsCatalog
)
from astropy import units as u


class TransmissiveElementTest(unittest.TestCase):

    def test_creation_from_transmittance_and_absorptance(self):
        """Test creation of TransmissiveElement from transmittance and absorptance"""
        t = Bandpass.flat(0.9)
        a = Bandpass.flat(0.05)
        te = TransmissiveElement(transmittance=t, absorptance=a)
        
        wv = te.waveset
        np.testing.assert_allclose(te.transmittance(wv), 0.9)
        np.testing.assert_allclose(te.absorptance(wv), 0.05)
        np.testing.assert_allclose(te.reflectance(wv), 0.05)

    def test_creation_from_transmittance_and_reflectance(self):
        """Test creation of TransmissiveElement from transmittance and reflectance"""
        t = Bandpass.flat(0.85)
        r = Bandpass.flat(0.10)
        te = TransmissiveElement(transmittance=t, reflectance=r)
        
        wv = te.waveset
        np.testing.assert_allclose(te.transmittance(wv), 0.85)
        np.testing.assert_allclose(te.reflectance(wv), 0.10)
        np.testing.assert_allclose(te.absorptance(wv), 0.05)

    def test_transmittance_in_band(self):
        """Test computation of average transmittance in a wavelength band"""
        t = Bandpass.flat(0.9)
        a = Bandpass.flat(0.1)
        te = TransmissiveElement(transmittance=t, absorptance=a)
        
        wv_band = (1000 * u.nm, 1500 * u.nm)
        t_mean, wv_min, wv_max = te.transmittance_in_band(wv_band, atol=(50, 50))
        
        self.assertAlmostEqual(t_mean, 0.9, places=5)
        self.assertIsInstance(wv_min, u.Quantity)
        self.assertIsInstance(wv_max, u.Quantity)


class TransmissiveSystemTest(unittest.TestCase):

    def setUp(self):
        """Create simple elements for testing"""
        self.t1 = Bandpass.flat(0.9)
        self.a1 = Bandpass.flat(0.05)
        self.element1 = TransmissiveElement(transmittance=self.t1, absorptance=self.a1)
        
        self.t2 = Bandpass.flat(0.95)
        self.a2 = Bandpass.flat(0.03)
        self.element2 = TransmissiveElement(transmittance=self.t2, absorptance=self.a2)
        
        self.t3 = Bandpass.flat(0.88)
        self.a3 = Bandpass.flat(0.02)
        self.element3 = TransmissiveElement(transmittance=self.t3, absorptance=self.a3)

    def test_add_single_element(self):
        """Test adding a single element to the system"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')
        
        self.assertEqual(len(ts.elements), 1)
        self.assertEqual(ts.elements[0]['name'], 'element1')
        self.assertEqual(ts.elements[0]['direction'], Direction.TRANSMISSION)

    def test_add_multiple_elements(self):
        """Test adding multiple elements to the system"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')
        ts.add(self.element2, Direction.TRANSMISSION, 'element2')
        ts.add(self.element3, Direction.REFLECTION, 'element3')
        
        self.assertEqual(len(ts.elements), 3)
        self.assertEqual(ts.elements[2]['direction'], Direction.REFLECTION)

    def test_remove_element(self):
        """Test removing an element from the system"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')
        ts.add(self.element2, Direction.TRANSMISSION, 'element2')
        ts.add(self.element3, Direction.TRANSMISSION, 'element3')
        
        ts.remove(1)
        
        self.assertEqual(len(ts.elements), 2)
        self.assertEqual(ts.elements[0]['name'], 'element1')
        self.assertEqual(ts.elements[1]['name'], 'element3')

    def test_transmittance_single_element(self):
        """Test transmittance computation for a single element"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')
        
        wv = ts.transmittance.waveset
        t = ts.transmittance(wv)
        
        np.testing.assert_allclose(t, 0.9, rtol=1e-5)

    def test_transmittance_multiple_elements(self):
        """Test transmittance computation for multiple elements"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')  # t=0.9
        ts.add(self.element2, Direction.TRANSMISSION, 'element2')  # t=0.95
        
        wv = ts.transmittance.waveset
        t = ts.transmittance(wv)
        
        expected_t = 0.9 * 0.95
        np.testing.assert_allclose(t, expected_t, rtol=1e-5)

    def test_transmittance_with_reflection(self):
        """Test transmittance computation with reflective elements"""
        # Element with reflection
        r_flat = Bandpass.flat(0.85)
        a_flat = Bandpass.flat(0.05)
        mirror = TransmissiveElement(reflectance=r_flat, absorptance=a_flat)
        
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')  # t=0.9
        ts.add(mirror, Direction.REFLECTION, 'mirror')  # r=0.85
        
        wv = ts.transmittance.waveset
        t = ts.transmittance(wv)
        
        expected_t = 0.9 * 0.85
        np.testing.assert_allclose(t, expected_t, rtol=1e-5)

    def test_transmittance_from_to(self):
        """Test partial transmittance computation using transmittance_from_to"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')  # t=0.9
        ts.add(self.element2, Direction.TRANSMISSION, 'element2')  # t=0.95
        ts.add(self.element3, Direction.TRANSMISSION, 'element3')  # t=0.88
        
        wv = ts.transmittance.waveset
        
        # Total transmittance
        t_total = ts.transmittance(wv)
        expected_total = 0.9 * 0.95 * 0.88
        np.testing.assert_allclose(t_total, expected_total, rtol=1e-5)
        
        # Transmittance from second element onwards
        t_from_1 = ts.transmittance_from_to(from_element=1)(wv)
        expected_from_1 = 0.95 * 0.88
        np.testing.assert_allclose(t_from_1, expected_from_1, rtol=1e-5)
        
        # Transmittance up to second element
        t_to_1 = ts.transmittance_from_to(to_element=1)(wv)
        expected_to_1 = 0.9 * 0.95
        np.testing.assert_allclose(t_to_1, expected_to_1, rtol=1e-5)
        
        # Transmittance of second element only
        t_only_1 = ts.transmittance_from_to(from_element=1, to_element=1)(wv)
        expected_only_1 = 0.95
        np.testing.assert_allclose(t_only_1, expected_only_1, rtol=1e-5)

    def test_element_idx_from_name(self):
        """Test retrieving element index by name"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'first')
        ts.add(self.element2, Direction.TRANSMISSION, 'second')
        ts.add(self.element3, Direction.TRANSMISSION, 'third')
        
        idx = ts.element_idx_from_name('second')
        self.assertEqual(idx, 1)
        
        idx = ts.element_idx_from_name('third')
        self.assertEqual(idx, 2)

    def test_add_transmissive_system(self):
        """Test adding a TransmissiveSystem to another TransmissiveSystem"""
        # Create first system
        ts1 = TransmissiveSystem()
        ts1.add(self.element1, Direction.TRANSMISSION, 'element1')
        ts1.add(self.element2, Direction.TRANSMISSION, 'element2')
        
        # Create second system
        ts2 = TransmissiveSystem()
        ts2.add(self.element3, Direction.TRANSMISSION, 'element3')
        
        # Add ts2 to ts1
        ts1.add(ts2)
        
        self.assertEqual(len(ts1.elements), 3)
        self.assertEqual(ts1.elements[2]['name'], 'element3')
        
        # Verify total transmittance
        wv = ts1.transmittance.waveset
        t = ts1.transmittance(wv)
        expected_t = 0.9 * 0.95 * 0.88
        np.testing.assert_allclose(t, expected_t, rtol=1e-5)

    def test_as_transmissive_element(self):
        """Test conversion of TransmissiveSystem to TransmissiveElement"""
        ts = TransmissiveSystem()
        ts.add(self.element1, Direction.TRANSMISSION, 'element1')
        ts.add(self.element2, Direction.TRANSMISSION, 'element2')
        
        te = ts.as_transmissive_element()
        
        self.assertIsInstance(te, TransmissiveElement)
        wv = te.waveset
        t = te.transmittance(wv)
        expected_t = 0.9 * 0.95
        np.testing.assert_allclose(t, expected_t, rtol=1e-5)


class TransmissiveSystemWithRealDataTest(unittest.TestCase):
    """Tests using real data from catalogs"""

    def test_system_with_glass_and_coating(self):
        """Test system composition with real glass and coating data"""
        # Load real elements
        glass = GlassesCatalog.infrasil_1mm_001()
        coating = CoatingsCatalog.ar_coating_broadband_001()
        
        # Create system
        ts = TransmissiveSystem()
        ts.add(coating, Direction.TRANSMISSION, 'AR_coating')
        ts.add(glass, Direction.TRANSMISSION, 'glass')
        ts.add(coating, Direction.TRANSMISSION, 'AR_coating_back')
        
        self.assertEqual(len(ts.elements), 3)
        
        # Verify transmittance is computable
        wv = ts.transmittance.waveset
        t = ts.transmittance(wv)
        
        self.assertTrue(np.all(t >= 0))
        self.assertTrue(np.all(t <= 1))
        
        # Transmittance with AR coating is less than glass alone
        # because we just multiply transmittances
        t_glass_only = glass.transmittance(wv)
        self.assertTrue(np.mean(t) < np.mean(t_glass_only))


if __name__ == "__main__":
    unittest.main()
