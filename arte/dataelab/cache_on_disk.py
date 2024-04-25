
import os
import pickle
import numbers
import tempfile
from functools import wraps

import numpy as np

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
    f.disk_cacher = DiskCacher(f)

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return f.disk_cacher.execute(self, *args, **kwargs)
    return wrapper


def set_tag(obj, tag):
    '''Setup tag on all DiskCacher objects inside this one, recursively'''
    for cacher in _discover_cachers(obj):
        cacher.set_tag(tag)


def clear_cache(obj):
    '''Clear cache on all DiskCacher objects inside this one, recursively'''
    for cacher in _discover_cachers(obj):
        cacher.clear_cache()


def set_tmpdir(obj, tmpdir):
    '''Set the temporary directory where cached data is stored'''
    for cacher in _discover_cachers(obj):
        cacher.set_tmpdir(tmpdir)


def set_prefix(obj, prefix):
    '''Set the prefix for each tag directory'''
    for cacher in _discover_cachers(obj):
        cacher.set_prefix(prefix)


def _discover_cachers(obj):
    '''Generator that yields all DiskCacher objects in all child members'''
    methods = {k: getattr(obj, k) for k in dir(obj) if callable(getattr(obj, k))}
    members = {k: getattr(obj, k) for k in dir(obj) if not callable(getattr(obj, k))}

    # Examine both methods and members, because DiskCacher replaces a method
    for _, method in methods.items():
        if hasattr(method, 'disk_cacher'):
            yield method.disk_cacher

    for name, member in members.items():
        if hasattr(member, 'disk_cacher'):
            yield member.disk_cacher

        # Recurse into all child members except for special variables.
        # For some reason, numeric types have infinite recursion
        if not name.startswith('__') and not isinstance(member, numbers.Number):
            yield from _discover_cachers(member)

#    What happens with properties?
#
#    properties = ({k: getattr(obj.__class__, k)
#                      for k in dir(obj.__class__)
#                      if isinstance(getattr(obj.__class__, k), property)})
#    methods.update(properties)


class DiskCacher():

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

    def _fullpath_no_extension(self):
        '''Full path of cache file on disk, without extension'''
        return os.path.join(self._tmpdir, self._prefix + self._tag, self._funcid)

    def _file_on_disk(self):
        '''Returns the cache file path, raises FileNotFoundError if not found'''
        extensions = ['.pickle', '.npy']
        for ext in extensions:
            path = self._fullpath_no_extension() + ext
            if os.path.exists(path):
                return path
        raise FileNotFoundError

    def _save_to_disk(self):
        fullpath = self._fullpath_no_extension()
        try:
            os.mkdir(os.path.join(self._tmpdir, self._prefix+self._tag))
        except FileExistsError:
            pass
        if isinstance(self._data, np.ndarray):
            np.save(fullpath+'.npy', self._data)
        else:
            with open(fullpath+'.pickle', 'wb') as f:
                pickle.dump(self._data, f)

    def _load_from_disk(self):
        path = self._file_on_disk()
        if path.endswith('.npy'):
            return np.load(path)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def _delete_from_disk(self):
        try:
            path = self._file_on_disk()
            os.unlink(path)
        except FileNotFoundError:
            pass

#




