# -*- coding: utf-8 -*-

import platform
import sys
import time
import unittest
import numpy as np
import multiprocessing as mp

from arte.utils.circular_buffer import NumpyCircularBuffer
from arte.utils.circular_buffer import SharedCircularBuffer


class CircularBufferTest(unittest.TestCase):

    def test_numpy(self):

        a = NumpyCircularBuffer(3, (2,), dtype=np.float32)
        data = np.arange(2)

        a.store(data)
        a.store(data + 1)
        a.store(data + 2)
        a.store(data + 3)

        np.testing.assert_array_equal(a.get(0), data + 3)
        np.testing.assert_array_equal(a.get(1), data + 1)
        assert a.position() == 1
        assert a.counter() == 4

    @unittest.skipIf(sys.version_info >= (3, 8),
                     "not compatible with python>3.8")
    @unittest.skipIf(platform.system() == 'Windows',
                     "not compatible with Windows")
    def test_data(self):
        '''
        Test that shared data can be passed back and forth
        to a task
        '''

        def task(arr):
            '''
            A task that polls on a trigger for max 5 seconds,
            and when triggered, modifies the input array.
            '''
            timeout = 5
            now = time.time()

            while True:
                if arr.counter() == 3:
                    break
                time.sleep(0.01)
                if time.time() - now >= timeout:
                    raise TimeoutError
            arr.store(np.ones(2,))

        arr = SharedCircularBuffer(3, (2,), dtype=np.float32)

        p = mp.Process(target=task, args=(arr,))
        p.start()

        data = np.arange(2)
        arr.store(data)
        arr.store(data + 1)
        arr.store(data + 2)  # This write unblocks the task
        p.join()  # Wait for the task to complete

        # Check that the task wrote its data
        np.testing.assert_array_equal(arr.get(0), np.ones((2,)))
        np.testing.assert_array_equal(arr.get(1), data + 1)
        assert arr.position() == 1
        assert arr.counter() == 4
