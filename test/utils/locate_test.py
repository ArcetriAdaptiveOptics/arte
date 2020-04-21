# -*- coding: utf-8 -*-

import os
import unittest
from unittest.mock import mock_open, patch

from arte.utils.locate import locate
from arte.utils.locate import locate_first
from arte.utils.locate import replace_in_file


# Replicate an example os.walk() output
_files = [('foo', ['bar, bar2'], ['foo.txt']),
          ('foo/bar', [], ['bar.txt']),
          ('foo/bar2', [], ['bar2.txt'])]

def _walk(rootdir):
    for f in _files:
        yield f
      
class LocateTest(unittest.TestCase):

    def test_replace_in_file(self):
        
        with patch('builtins.open', mock_open(read_data='123foo456foo')) as m:
             replace_in_file('nome', 'foo', 'bar')
            
        m().write.assert_called_with('123bar456bar')
   
    def test_locate(self):
                
        with patch.object(os, 'walk', _walk):
             files = list(locate('*r2.txt'))
             
        assert len(files)==1
        assert files[0] == 'foo/bar2/bar2.txt'

    def test_locate_first(self):
                
        with patch.object(os, 'walk', _walk):
             file = locate_first('*r2.txt')
        assert file == 'foo/bar2/bar2.txt'

        with patch.object(os, 'walk', _walk):
             file = locate_first('*abc')
        assert file == None
