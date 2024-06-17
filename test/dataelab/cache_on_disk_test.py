# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
import numpy as np
import astropy.units as u
from functools import cached_property

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
        set_tag(self, '2222')
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


class ClassNoTag():
    def __init__(self):
        super().__init__()
    
    @cache_on_disk
    def a_method(self, foo):
        pass


class NumpyCache(Parent):
    def __init__(self, tag):
        set_tag(self, tag)

    @cache_on_disk
    def a_method(self, arg):
        return arg*2


class ClassWithProperties():

    def __init__(self):
        self.total = 0

    def I_am_a_method(self):
        '''We will check that self.total is NOT incremented by 1'''
        self.total += 1
        return 1

    @property
    def I_am_a_property(self):
        '''We will check that self.total is NOT incremented by 2'''
        self.total += 2
        return 2

    @cached_property
    def I_am_a_cached_property(self):
        '''We will check that self.total is NOT incremented by 4'''
        self.total += 4
        return 4


class CacheOnDiskTest(unittest.TestCase):

    def setUp(self):
        self.ee = Child(tag='1234')

    def test_parent_path_name(self):
        '''If a method is not redefined, we want the child class name anyway'''
        fullpath = get_disk_cacher(self.ee, self.ee.a_method).fullpath()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'root.Parent.a_method.npy')

    def test_reimplemented_path_name(self):
        '''If both parent and child use @cache_on_disk on the same method, we want the child class name'''
        fullpath = get_disk_cacher(self.ee, self.ee.b_method).fullpath()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'root.Child.b_method.npy')

    def test_child_path_name(self):
        '''If a child redefines a method using @cache_on_disk, we want the child class name'''
        fullpath = get_disk_cacher(self.ee, self.ee.c_method).fullpath()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'root.Child.c_method.npy')

    def test_force_parent_path_name(self):
        '''For a redefined method, using the parent class name explictly results in the parent class name'''
        p = Parent()
        fullpath = get_disk_cacher(p, p.b_method).fullpath()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache2222', 'root.Parent.b_method.npy')

    def test_member_path_name(self):
        '''If a method is defined in a class used as attribute, we want that attribute class name'''
        fullpath = get_disk_cacher(self.ee.foo, self.ee.foo.my_method).fullpath()
        assert fullpath == os.path.join(tempfile.gettempdir(), 'cache1234', 'root.foo.Member.my_method.npy')

    def test_parent_method(self):
        '''Test caching of parent method and cache clearing'''
        assert self.ee.a_method(3,2) == 5
        fname = get_disk_cacher(self.ee, self.ee.a_method).fullpath()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_reimplemented_method(self):
        '''Test caching of parent method and cache clearing'''
        assert self.ee.b_method(3,2) == 1.5
        fname = get_disk_cacher(self.ee, self.ee.b_method).fullpath()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_child_method(self):
        '''Test caching of child method and cache clearing'''
        assert self.ee.c_method(3,2) == 1.5
        fname = get_disk_cacher(self.ee, self.ee.c_method).fullpath()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_member_method(self):
        '''Test caching of member method and cache clearing'''
        assert self.ee.foo.my_method(3,2) == 5
        fname = get_disk_cacher(self.ee.foo, self.ee.foo.my_method).fullpath()
        assert os.path.exists(fname)
        clear_cache(self.ee)
        assert not os.path.exists(fname)

    def test_non_cached_method(self):
        '''Non-cached methods do not have a disk_cacher attribute'''
        assert self.ee.d_method(3,2) == 6
        with self.assertRaises(AttributeError):
            _ = get_disk_cacher(self.ee, self.ee.d_method).fullpath()

    def test_cache_hit(self):
        '''Test that we are actually hitting the disk cache'''
        myee = Child(tag='4321')
        clear_cache(myee)
        assert myee.c_method_counter == 0
        assert myee.c_method(3,2) == 1.5
        assert myee.c_method_counter == 1
        assert myee.c_method(3,2) == 1.5   # cache hit
        assert myee.c_method_counter == 1  # no counter increase
        fname = get_disk_cacher(myee, myee.c_method).fullpath()
        clear_cache(myee)
        assert not os.path.exists(fname)
        assert myee.c_method(3,2) == 1.5   # recalc
        assert myee.c_method_counter == 2  # counter increase
        fname = get_disk_cacher(myee, myee.c_method).fullpath()
        assert os.path.exists(fname)       # File is there
        clear_cache(myee)
        assert not os.path.exists(fname)   # And no more.

    def test_set_prefix(self):
        '''Test prefix change'''
        myee = Child(tag='4321')
        orig_prefix = get_disk_cacher(myee, myee.a_method)._prefix
        set_prefix(myee, 'prefix')
        fullpath = get_disk_cacher(myee, myee.a_method).fullpath()
        set_prefix(myee, orig_prefix)
        assert fullpath == os.path.join(tempfile.gettempdir(), 'prefix4321', 'root.Parent.a_method.npy')

    def test_set_tmpdir(self):
        '''Test tempdir change'''
        myee = Child(tag='4321')
        set_tmpdir(myee, 'another_tempdir')
        fullpath = get_disk_cacher(myee, myee.a_method).fullpath()
        assert fullpath == os.path.join('another_tempdir', 'cache4321', 'root.Parent.a_method.npy')

    def test_notag(self):
        '''Test that an exception is raised when no tag has been set'''
        ee_notag = ClassNoTag()
        with self.assertRaises(Exception):
            _ = get_disk_cacher(ee_notag, ee_notag.a_method).fullpath()

    def test_automatic_creation(self):
        '''Test that DiskCacher instances are created automatically whenever accessed'''
        ee_notag = ClassNoTag()
        set_tmpdir(ee_notag, '/tmp')   # Must not raise

    def test_numpy(self):
        ee = NumpyCache(tag='testnumpy')
        clear_cache(ee)
        _ = ee.a_method(np.ones(1))
        np.testing.assert_array_equal(ee.a_method(None), np.ones(1)*2)
        clear_cache(ee)

    def test_numpyfile_exists(self):
        ee = NumpyCache(tag='testnumpy_exists')
        clear_cache(ee)
        _ = ee.a_method(np.ones(1))
        assert os.path.exists(get_disk_cacher(ee, ee.a_method).fullpath())
        clear_cache(ee)

    def test_pickle(self):
        ee = NumpyCache(tag='testpickle')
        clear_cache(ee)
        _ = ee.a_method(np.ones(1) * u.m)
        np.testing.assert_array_equal(ee.a_method(None), np.ones(1)*2*u.m)
        clear_cache(ee)

    def test_picklefile_exists(self):
        ee = NumpyCache(tag='testpickle_exists')
        clear_cache(ee)
        _ = ee.a_method('a_string')
        assert os.path.exists(get_disk_cacher(ee, ee.a_method).fullpath())
        clear_cache(ee)

    def test_no_property_trigger(self):
        ee = ClassWithProperties()
        set_tag(ee, 'foo')
        assert ee.total == 0

# __oOo__
