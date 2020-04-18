# -*- coding: utf-8 -*-

#!/usr/bin/env python
import shutil
import unittest
from arte.utils.compareIDL import compareIDL

# Tests will be run only if IDL is installed
have_idl = (shutil.which('idl') != None)

def strip_all(lst):
    return [x.strip() for x in lst]

idlscript = '''
a=!pi
b=1
'''
        
pythonscript = '''
import math
a=math.pi
b=2
'''


class CompareIDLTest(unittest.TestCase):

    @unittest.skipUnless(have_idl, 'IDL is not installed')
    def test_compare(self):
        
        assert(compareIDL(idlscript, pythonscript, ['a']) == True)

    @unittest.skipUnless(have_idl, 'IDL is not installed')
    def test_fail(self):

        assert(compareIDL(idlscript, pythonscript, ['b']) == False)


if __name__ == "__main__":
    unittest.main()
