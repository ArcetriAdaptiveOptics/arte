#!/usr/bin/env python
import doctest
import unittest
import numpy as np

import matplotlib
matplotlib.use('Agg')   # Draw in background
import matplotlib.pyplot as plt

from arte.utils.show_movie import show_movie


class ShowMovieTest(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_docstring(self):
        import arte.utils.show_movie as show_movie_module
        doctest.testmod(show_movie_module, raise_on_error=True)

    def test_cube(self):
        cube = np.random.rand(5, 8, 8)
        ani = show_movie(cube, show=False)
        frames = ani._framedata
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0][0].get_clim(), (cube.min(), cube.max()))

    def test_list_of_frames(self):
        frames = [np.random.rand(8, 8) for _ in range(3)]
        ani = show_movie(frames, show=False)
        self.assertEqual(len(ani._framedata), 3)

    def test_offset(self):
        cube = np.ones((3, 8, 8))
        ani = show_movie(cube, offset=np.ones((8, 8)), show=False)
        np.testing.assert_array_equal(ani._framedata[0][0].get_array(), np.zeros((8, 8)))

    def test_wrong_input(self):
        with self.assertRaises(TypeError):
            show_movie(np.random.rand(8, 8), show=False)
        with self.assertRaises(TypeError):
            show_movie([np.zeros((8, 8)), np.zeros((4, 4))], show=False)
        with self.assertRaises(TypeError):
            show_movie(np.random.rand(3, 8, 8), offset=np.zeros(8), show=False)
        with self.assertRaises(TypeError):
            show_movie(np.random.rand(3, 8, 8), offset=np.zeros((4, 4)), show=False)


if __name__ == "__main__":
    unittest.main()
