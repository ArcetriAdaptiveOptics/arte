# -*- coding: utf-8 -*-

import time
import unittest
import numpy as np
import multiprocessing as mp
from itertools import repeat

from arte.utils.shared_array import SharedArray


class SharedArrayTest(unittest.TestCase):

    def test_data(self):
        '''
        Test that shared data can be passed back and forth
        to a task
        '''
        def task(arr, trig):
            '''
            A task that polls on a trigger for max 5 seconds,
            and when triggered, modifies the input array.
            '''
            timeout=5
            now=time.time()

            while True:
                if trig[0]==1:
                    break
                time.sleep(0.01)
                if time.time()-now >= timeout:
                    raise TimeoutError
            arr[1] = arr[0]+1
    
        arr = SharedArray((2,), np.int32)
        trig = SharedArray((1,), np.int32)
    
        p = mp.Process(target = task, args=(arr, trig))
        p.start()
    
        arr[0]=1    # Initialize some data
        trig[0]=1   # Trigger the task
        p.join()    # Wait for the task to complete
        
        assert arr[1] == 2   # Check task result

    @unittest.skip("Does not work with a Pool unless an mp.Manager is used")    
    def test_pool(self):
        '''
        Test that we can read back from a pool
        using a SharedArray.
        '''

        def pool_task(n, arr):
            arr[n] = n

        arr = SharedArray((10,), np.int32)

        with mp.Pool(4) as p:
            p.map(pool_task, zip(range(10), repeat(arr,10)))
        
        np.testing.assert_array_equal( arr[:], np.arange(10, dtype=np.int32)) 