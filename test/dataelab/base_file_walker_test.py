import unittest
from arte.dataelab.base_file_walker import AbstractFileNameWalker

class BaseFileWalkerTest(unittest.TestCase):

    def test_abstract(self):
        '''Test that a derived class must implement abstract methods'''
        class TestFileWalker(AbstractFileNameWalker):
            pass
        with self.assertRaises(TypeError):
            _ = TestFileWalker()


if __name__ == "__main__":
    unittest.main()
