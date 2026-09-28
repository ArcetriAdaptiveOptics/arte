# -*- coding: utf-8 -*-

import re
import unittest

from arte.dataelab.tag import Tag


class TagTest(unittest.TestCase):

    def test_underscore_tag(self):
        tag = Tag('20240101_120000')
        self.assertEqual(str(tag), '20240101_120000')
        self.assertEqual(tag.get_day_as_string(), '20240101')
        # The whole tag is repeated as remainder
        self.assertEqual(tag.get_remainder_as_string(), '20240101_120000')

    def test_slash_tag(self):
        tag = Tag('20240101/120000')
        self.assertEqual(tag.get_day_as_string(), '20240101')
        self.assertEqual(tag.get_remainder_as_string(), '120000')

    def test_prefixed_tag(self):
        tag = Tag('KAPA20240101_120000')
        self.assertEqual(tag.get_day_as_string(), 'KAPA20240101')

    def test_invalid_tags(self):
        for s in ['20240101120000', '2024_0101_120000', '2024/01/01', '20240101_12/00']:
            with self.subTest(tag=s):
                with self.assertRaises(AssertionError):
                    Tag(s)

    def test_create_tag(self):
        tag = Tag.create_tag()
        self.assertRegex(str(tag), re.compile(r'^\d{8}_\d{6}$'))


if __name__ == "__main__":
    unittest.main()
