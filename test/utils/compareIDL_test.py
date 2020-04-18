# -*- coding: utf-8 -*-

#!/usr/bin/env python
import shutil
import unittest
from arte.utils.compareIDL import compareIDL

# Tests will be run only if IDL is installed
have_idl = (shutil.which('idl') != None)

def strip_all(lst):
    return [x.strip() for x in lst]

class CompareIDLTest(unittest.TestCase):

    def setUp(self):
        
        self.idlscript = '''
        a=!pi
        b=1
        '''
        
        # Python does not like leading whitespace.
        self.pythonscript = '''
import math
a=math.pi
b=2
        '''

    @unittest.skipUnless(have_idl, 'IDL is not installed')
    def test_compare(self):
        
        assert(compareIDL(self.idlscript, self.pythonscript, ['a']) == True)

    @unittest.skipUnless(have_idl, 'IDL is not installed')
    def test_fail(self):

        assert(compareIDL(self.idlscript, self.pythonscript, ['b']) == False)


if __name__ == "__main__":
    unittest.main()
