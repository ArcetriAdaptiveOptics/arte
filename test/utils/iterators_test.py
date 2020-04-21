# -*- coding: utf-8 -*-

#!/usr/bin/env python
import doctest
import unittest

class IteratorsTest(unittest.TestCase):

    def test_docstrings(self):
        from arte.utils import iterators
        doctest.testmod(iterators, raise_on_error=True)