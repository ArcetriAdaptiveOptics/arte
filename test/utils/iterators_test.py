# -*- coding: utf-8 -*-

#!/usr/bin/env python
import doctest
import unittest
from arte.utils.iterators import flatten, pairwise


class IteratorsTest(unittest.TestCase):

    def test_flatten(self):
        doctest.run_docstring_examples(flatten, locals())
        
    def test_pairwise(self):
        doctest.run_docstring_examples(pairwise, locals())