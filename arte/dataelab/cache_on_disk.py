
import os
import pickle
import tempfile
from functools import wraps
from collections import defaultdict

import numpy as np
import astropy.units as u


class TagNotSetException(Exception):
    '''Cache tag has not been set'''


def cache_on_disk(f):
    '''Persistent disk cache for method return values.

    Only use this decorator for methods whose return value never changes.

    Methods decorated with cache_on_disk() will write their return value into
    a npy or pickle file depending on the value type. Subsequent method
    calls will reuse the previous return value, stored in a memory buffer.
    New instances using the same tag will read data from disk on the first call.

    Caching only starts after a global tag has been defined. Call
    set_tag(obj, tag) with obj set the highest object in the hierarchy
    (typically the analyzer instance) to initialize disk caching for all
    child objects/methods.

    Tags should be uniquely identifying the dataset to be stored.
    One suggestion is to use both a system identifier (like KAPA or LBTISX)
    and a timestamp, for example LBTISX20240410_112233.

    Call clear_cache(obj) to delete all temporary files for all child
    objects/methods. The methods code will be run and stored again
    when called.

    Data is stored in tmpdir/prefix<tag>/file.npy, where:
     - tmpdir is by default the system temporary directory
     - prefix is by default "cache"
     - tag must be set by the owner class by calling set_tag at some point

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
    for instance, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].set_tag(tag)


def clear_cache(root_obj):
    '''Clear cache for root_obj and child/member objects'''
    for instance, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].clear_cache()


def set_tmpdir(root_obj, tmpdir):
    '''Set the directory where cached data is stored'''
    for instance, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].set_tmpdir(tmpdir)


def set_prefix(root_obj, prefix):
    '''Set the prefix for each tag directory'''
    for instance, cacher_list in _discover_cachers(root_obj):
        cacher_list[instance].set_prefix(prefix)


def _discover_cachers(obj, seen=None):
    '''Generator that yields all DiskCacher objects in all child members'''

    # This set is used to avoid infinite recursion
    # when objects have circular references
    if seen is None:
        seen = set()

    # Some packages like astropy.units define attributes
    # that just raise exceptions if they are accessed (!)
    valid_attrs = {}
    for name in dir(obj):
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
            yield obj, method.cacher_list

    for name, member in members.items():
        if hasattr(member, 'cacher_list'):
            yield obj, member.cacher_list

        # Recurse into all child members except for special variables.
        if (not name.startswith('__') and not type(member) in seen):
            seen.update([type(member)])
            yield from _discover_cachers(member, seen=seen)

#    What happens with properties?
#
#    properties = ({k: getattr(obj.__class__, k)
#                      for k in dir(obj.__class__)
#                      if isinstance(getattr(obj.__class__, k), property)})
#    methods.update(properties)


class DiskCacher():
    '''Class implementing a persistent disk cache'''

    def __init__(self, f):
        self._tag = None
        self._data = None
        self._tmpdir = tempfile.gettempdir()
        self._original_function = f
        self._funcid = f.__qualname__
        self._prefix = 'cache'

    def set_tag(self, tag):
        '''Unique tag identifying the Elab instance'''
        self._tag = tag

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

    def execute(self, *args, **kwargs):
        '''Cache lookup'''
        if self._tag is not None:
            # Local memory cache
            if self._data is not None:
                return self._data
            # Disk cache
            try:
                self._data = self._load_from_disk()
                return self._data
            except FileNotFoundError:
                self._data = self._original_function(*args, **kwargs)
                self._save_to_disk()
                return self._data
        else:
            return self._original_function(*args, **kwargs)

    def fullpath_no_extension(self):
        '''Full path of cache file on disk, without extension'''
        if self._tag is None:
            raise TagNotSetException('Disk cache tag has not been set')
        return os.path.join(self._tmpdir, self._prefix + self._tag, self._funcid)

    def file_on_disk(self):
        '''Returns the cache file path, raises FileNotFoundError if not found'''
        extensions = ['.pickle', '.npy']
        for ext in extensions:
            path = self.fullpath_no_extension() + ext
            if os.path.exists(path):
                return path
        raise FileNotFoundError

    def _save_to_disk(self):
        fullpath = self.fullpath_no_extension()
        try:
            os.mkdir(os.path.join(self._tmpdir, self._prefix+self._tag))
        except FileExistsError:
            pass
        if isinstance(self._data, np.ndarray):
            if isinstance(self._data, u.Quantity):
                np.save(fullpath+'.npy', self._data.value)
                with open(fullpath+'.npy.unit.pickle', 'wb') as f:
                    pickle.dump(self._data.unit, f)
            else:
                np.save(fullpath+'.npy', self._data)
        else:
            with open(fullpath+'.pickle', 'wb') as f:
                pickle.dump(self._data, f)

    def _load_from_disk(self):
        path = self.file_on_disk()
        if path.endswith('.npy'):
            data = np.load(path)
            print('Loading from disk')
            if os.path.exists(path+'.unit.pickle'):
                with open(path+'.unit.pickle', 'rb') as f:
                    unit = pickle.load(f)
                data = data * unit
            return data
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _delete_from_disk(self):
        extensions = ['.pickle', '.npy', 'unit.pickle']
        for ext in extensions:
            path = self.fullpath_no_extension() + ext
            if os.path.exists(path):
                os.unlink(path)

# __oOo__