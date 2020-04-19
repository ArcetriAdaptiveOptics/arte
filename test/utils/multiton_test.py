# -*- coding: utf-8 -*-

import unittest

from arte.utils.multiton import multiton, multiton_id

class MultitonTest(unittest.TestCase):

    @multiton
    class AMultiton():
        def __init__(self,a,b):
            self.a = a
            self.b = b

    @multiton_id
    class AnIdMultiton():
        def __init__(self,a,b):
            self.a = a
            self.b = b
    
    def test_multiton(self):
        
        a1 = self.AMultiton(1,2)
        a2 = self.AMultiton(1,2)
        b = self.AMultiton(3,4)
        
        assert a1 is a2
        assert a1 is not b

    def test_multiton_id(self):
        
        a1 = self.AnIdMultiton(1,2)
        a2 = self.AnIdMultiton(1,2)
        b = self.AnIdMultiton(3,4)
        
        assert a1 is a2
        assert a1 is not b
        
        # When passing a mutable type, multiton_id is not fazed
        dd = {'a':1, 'b':2}
        c = self.AnIdMultiton(dd, 2)
        e = self.AnIdMultiton(dd, 1)
        
        dd['b'] = 3
        d = self.AnIdMultiton(dd, 2)
        
        assert c is d
        assert c is not e
        

        
if __name__ == "__main__":
    unittest.main()

