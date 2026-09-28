# -*- coding: utf-8 -*-

import unittest

from arte.dataelab.base_indexer import BaseIndexer


class BaseIndexerTest(unittest.TestCase):
    '''Behaviour of BaseIndexer with the default keywords.

    Custom keywords are broken, see known_defects_test.py (A2).
    '''

    def setUp(self):
        self.indexer = BaseIndexer()

    def test_default_is_all_elements(self):
        self.assertEqual(self.indexer.process_args(), slice(None, None))

    def test_single_element(self):
        self.assertEqual(self.indexer.process_args(element=5), 5)

    def test_list_of_elements(self):
        self.assertEqual(self.indexer.process_args(elements=[1, 5, 9]), [1, 5, 9])

    def test_range(self):
        self.assertEqual(self.indexer.process_args(from_element=2, to_element=7), slice(2, 7))
        self.assertEqual(self.indexer.process_args(first=2, last=7), slice(2, 7))

    def test_open_range(self):
        self.assertEqual(self.indexer.process_args(from_element=2), slice(2, None))
        self.assertEqual(self.indexer.process_args(to_element=7), slice(None, 7))

    def test_positional_args_are_returned_as_tuple(self):
        # Current behaviour, relied upon by TimeSeries._index_data() which
        # interprets a tuple as a per-dimension index.
        self.assertEqual(self.indexer.process_args(5), (5,))
        self.assertEqual(self.indexer.process_args(1, 2), (1, 2))

    def test_positional_args_take_precedence(self):
        self.assertEqual(self.indexer.process_args(3, element=5), (3,))

    def test_unknown_keywords_are_ignored(self):
        self.assertEqual(self.indexer.process_args(foo=3), slice(None, None))


if __name__ == "__main__":
    unittest.main()
