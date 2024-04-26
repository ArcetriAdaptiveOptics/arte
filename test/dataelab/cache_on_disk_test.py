# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

from arte.dataelab.cache_on_disk import cache_on_disk, set_tag, set_tmpdir
from arte.dataelab.cache_on_disk import set_prefix, clear_cache, get_disk_cacher


class Member():
    @cache_on_disk
    def my_method(self, arg1, arg2):
        '''Cached method to be composed'''
        return arg1 +  arg2


class Parent():
    def __init__(self):
        '''Child object test'''
        self.foo = Member()

    @cache_on_disk
    def a_method(self, arg1, arg2):
        '''Cached method, not reimplemented in child'''
        return arg1 + arg2

    @cache_on_disk
    def b_method(self, arg1, arg2):
        '''Cached method,  reimplemented in child'''
        return arg1 - arg2

    def c_method(self, arg1, arg2):
        '''Non-cached method, reimplemted and cached in child'''
        return arg1 * arg2

    def d_method(self, arg1, arg2):
        '''Non-cached method, not reimplemented in child'''
        return arg1 * arg2


class Child(Parent):
    def __init__(self, tag):
        super().__init__()
        set_tag(self, tag)
        self.c_method_counter = 0

    @cache_on_disk
    def b_method(self, arg1, arg2):
        '''Reimplemented method'''
        return arg1 / arg2

    @cache_on_disk
    def c_method(self, arg1, arg2):
        self.c_method_counter += 1
        return arg1 / arg2


class ChildNoTag(Parent):
    def __init__(self):
        super().__init__()
        self.c_method_counter = 0


class CacheOnDiskTest(unittest.TestCase):

    def setUp(self):
        self.ee = Child(tag='1234')
        self.ee_notag = ChildNoTag()

    def test_parent_path_name(self):
        '''If a method is not redefined, we want the parent's class name'''
        fullpath = get_disk_cacher(self.ee, self.ee.a_method).fullpath_no_extension()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'Parent.a_method')

    def test_reimplemented_path_name(self):
        '''If both parent and child use @cache_on_disk on the same method, we want the child's class name'''
        fullpath = get_disk_cacher(self.ee, self.ee.b_method).fullpath_no_extension()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'Child.b_method')

    def test_child_path_name(self):
        '''If a child redefines a method using @cache_on_disk, we want the child's class name'''
        fullpath = get_disk_cacher(self.ee, self.ee.c_method).fullpath_no_extension()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'Child.c_method')

    def test_member_path_name(self):
        '''If a method is defined in a class used as attribute, we want that attribute class name'''
        fullpath = get_disk_cacher(self.ee.foo, self.ee.foo.my_method).fullpath_no_extension()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'Member.my_method')

    def test_parent_method(self):
        '''Test caching of parent method and cache clearing'''
        assert self.ee.a_method(3,2) == 5
        fname = get_disk_cacher(self.ee, self.ee.a_method).file_on_disk()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_reimplemented_method(self):
        '''Test caching of parent method and cache clearing'''
        assert self.ee.b_method(3,2) == 1.5
        fname = get_disk_cacher(self.ee, self.ee.b_method).file_on_disk()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_child_method(self):
        '''Test caching of child method and cache clearing'''
        assert self.ee.c_method(3,2) == 1.5
        fname = get_disk_cacher(self.ee, self.ee.c_method).file_on_disk()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_member_method(self):
        '''Test caching of member method and cache clearing'''
        assert self.ee.foo.my_method(3,2) == 5
        fname = get_disk_cacher(self.ee.foo, self.ee.foo.my_method).file_on_disk()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_non_cached_method(self):
        '''Non-cached methods do not have a disk_cacher attribute'''
        assert self.ee.d_method(3,2) == 6
        with self.assertRaises(AttributeError):
            _ = get_disk_cacher(self.ee, self.ee.d_method).file_on_disk()

    def test_cache_hit(self):
        '''Test that we are actually hitting the disk cache'''
        myee = Child(tag='4321')
        clear_cache(myee)
        assert myee.c_method_counter == 0
        assert myee.c_method(3,2) == 1.5
        assert myee.c_method_counter == 1
        assert myee.c_method(3,2) == 1.5   # cache hit
        assert myee.c_method_counter == 1  # no counter increase
        fname = get_disk_cacher(myee, myee.c_method).file_on_disk()
        clear_cache(myee)
        assert not os.path.exists(fname)
        assert myee.c_method(3,2) == 1.5   # recalc
        assert myee.c_method_counter == 2  # counter increase
        fname = get_disk_cacher(myee, myee.c_method).file_on_disk()
        assert os.path.exists(fname)       # File is there
        clear_cache(myee)
        assert not os.path.exists(fname)   # And no more.

    def test_set_prefix(self):
        '''Test prefix change'''
        myee = Child(tag='4321')
        orig_prefix = get_disk_cacher(myee, myee.a_method)._prefix
        set_prefix(myee, 'prefix')
        fullpath = get_disk_cacher(myee, myee.a_method).fullpath_no_extension()
        set_prefix(myee, orig_prefix)
        assert fullpath == os.path.join(tempfile.gettempdir(), 'prefix4321', 'Parent.a_method')

    def test_set_tmpdir(self):
        '''Test tempdir change'''
        myee = Child(tag='4321')
        set_tmpdir(myee, 'another_tempdir')
        fullpath = get_disk_cacher(myee, myee.a_method).fullpath_no_extension()
        assert fullpath == os.path.join('another_tempdir', 'cache4321', 'Parent.a_method')

    def test_notag(self):
        '''Test that an exception is raised when no tag has been set'''
        with self.assertRaises(Exception):
            _ = get_disk_cacher(self.ee_notag, self.ee_notag.a_method).fullpath_no_extension()

# __oOo__