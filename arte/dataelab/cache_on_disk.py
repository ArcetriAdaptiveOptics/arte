'''Persistent disk cache for immutable method return values.

Disk cache in the style of LBT elab_lib
'''

import os
import pickle
import logging
import tempfile
from functools import wraps, cached_property
from collections import defaultdict
from types import ModuleType

import numpy as np


class TagNotSetException(Exception):
    '''Cache tag has not been set'''


def cache_on_disk(f):
    '''Persistent disk cache for immutable method return values.

    Only use this decorator for methods whose return value never changes.

    Methods decorated with cache_on_disk() will write their return value into
    a npy or pickle file depending on the value type. Subsequent method
    calls will reuse the previous return value, stored in a memory buffer.
    New instances using the same tag will read data from disk on the first call.

    Caching only starts after a global tag has been defined. Call
    set_tag(obj, tag) with obj set the highest object in the hierarchy
    (typically the analyzer instance) to initialize disk caching for all
    inherited and composed (via attributes) objects.

    Tags should be uniquely identifying the dataset to be stored.
    One suggestion is to use both a system identifier (like KAPA or LBTISX)
    and a timestamp, for example LBTISX20240410_112233.

    Call clear_cache(obj) to delete all temporary files for all inherited
    and composed (via attributes) objects. The methods code will be run
    and stored again when called.

    Data is stored in tmpdir/prefix<tag>/filename.npy, where:
     - tmpdir is by default the system temporary directory
     - prefix is by default "cache"
     - tag must be set by the owner class by calling set_tag() at some point
     - "filename" identifies the method and class name

    File always has extension ".npy". Any type different from a numpy array will
    be pickled instead, using the same extension and leveraging numpy's
    transparent object pickling.

    Defaults for tmpdir and prefix can be overriden using set_tmpdir() and
    set_prefix(), each with two arguments:
     - the highest object in the hiearchy (typically the anayzer instance)
     - the new value for tmpdir or prefix
    '''

    # Bind this function to a DiskCacher constructor
    def new_cacher():
        return DiskCacher(f)

    # Access a DiskCacher based on the instance
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return wrapper.cacher_list[self].execute(self, *args, **kwargs)

    # Build a new DiskCacher whenever a new instance accesses this method
    wrapper.cacher_list = defaultdict(new_cacher)
    return wrapper


def get_disk_cacher(instance, method):
    '''Return the DiskCacher used by *method* for instance *instance*'''
    return method.cacher_list[instance]


def set_tag(root_obj, tag):
    '''Setup disk cache tag for root_obj and child/member objects'''
    for instance, instance_name, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].set_tag(tag, instance_name)


def clear_cache(root_obj):
    '''Clear cache for root_obj and child/member objects'''
    for instance, _, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].clear_cache()


def set_tmpdir(root_obj, tmpdir):
    '''Set the directory where cached data is stored'''
    for instance, _, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].set_tmpdir(tmpdir)


def set_prefix(root_obj, prefix):
    '''Set the prefix for each tag directory'''
    for instance, _, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].set_prefix(prefix)


_logger = None

def set_logfile(filename, level=logging.DEBUG, name='cache_on_disk'):
    '''Set the file where log output will be written'''
    global _logger
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(message)s',
                                datefmt='%Y-%m-%d:%H:%M:%S')
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)


def _discover_cachers(obj, objname='root', seen=None):
    '''Generator that yields all DiskCacher objects in all child members'''

    # This set is used to avoid infinite recursion
    # when objects have circular references
    if seen is None:
        seen = set()

    # Exclude properties and cached properties
    properties = ([k for k in dir(obj.__class__)
                   if isinstance(getattr(obj.__class__, k), (property, cached_property))])

    # Some packages like astropy.units define attributes
    # that just raise exceptions if they are accessed (!)
    valid_attrs = {}
    for name in dir(obj):
        if not name in properties:
            try:
                valid_attrs[name] = getattr(obj, name)
            except AttributeError:
                pass

    methods = {k: v for k, v in valid_attrs.items() if callable(v)}
    members = {k: v for k, v in valid_attrs.items() if not callable(v)}

    # Examine both methods and members, because DiskCacher replaces a method
    for name, method in methods.items():
        seen.update([type(method)])
        if hasattr(method, 'cacher_list'):
            yield obj, objname, method.cacher_list

    for name, member in members.items():
        if hasattr(member, 'cacher_list'):
            yield obj, objname, member.cacher_list

        # Recurse into all child members except for special variables.
        if not name.startswith('__') and not type(member) in seen:
            seen.update([type(member)])
            if not isinstance(member, ModuleType):
                yield from _discover_cachers(member, objname + '.' + name, seen=seen)


class DiskCacher():
    '''Class implementing a persistent disk cache'''

    def __init__(self, f):
        self._tag = None
        self._data = None
        self._tmpdir = tempfile.gettempdir()
        self._original_function = f
        self._method_name = f.__qualname__
        self._funcid = None
        self._prefix = 'cache'

    def set_tag(self, tag, instance_name):
        '''Set the tag used to make cached data persistent'''
        self._tag = tag
        self._funcid = instance_name + '.' + self._method_name

    def set_tmpdir(self, tmpdir):
        '''Set the directory where cached data is stored'''
        self._tmpdir = tmpdir

    def set_prefix(self, prefix):
        '''Set prefix for tag directories'''
        self._prefix = prefix

    def clear_cache(self):
        '''Remove cache from disk and memory'''
        self._data = None
        if self._tag:
            self._delete_from_disk()

    def _log(self, message):
        if _logger:
            _logger.debug(' '.join((self._tag, self._funcid, message)))

    def execute(self, *args, **kwargs):
        '''Cache lookup'''
        if self._tag is not None:
            # Local memory cache
            if self._data is not None:
                self._log('Internal cache hit')
                return self._data
            # Disk cache
            try:
                self._data = self._load_from_disk()
                return self._data
            except FileNotFoundError as e:
                self._log('Exception reading from disk: '+str(e))
                self._log('Calling function to cache')
                self._data = self._original_function(*args, **kwargs)
                self._save_to_disk()
                return self._data
        else:
            return self._original_function(*args, **kwargs)

    def fullpath(self):
        '''Full path of cache file on disk, without extension'''
        if self._tag is None:
            raise TagNotSetException('Disk cache tag has not been set')
        if self._funcid is None:
            raise TagNotSetException('Function ID has not been set')
        return os.path.join(self._tmpdir, self._prefix + self._tag, self._funcid + '.npy')

    def _save_to_disk(self):
        self._log('Saving ' + self.fullpath())
        try:
            os.mkdir(os.path.join(os.path.dirname(self.fullpath())))
        except FileExistsError:
            pass

        try:
            # Use exact type match because e.g. u.Quantity is a subclass
            # but cannot be used with np.save
            if type(self._data) == np.ndarray:          # pylint: disable=C0123
                np.save(self.fullpath(), self._data)
            else:
                pickle.dump(self._data, open(self.fullpath(), 'wb'))
        except Exception as e:
            self._log('Exception saving to disk: '+str(e))

    def _load_from_disk(self):
        self._log('Reading ' + self.fullpath())
        return np.load(self.fullpath(), allow_pickle=True)

    def _delete_from_disk(self):
        self._log('Deleting ' + self.fullpath())
        if os.path.exists(self.fullpath()):
            os.unlink(self.fullpath())

# __oOo__
