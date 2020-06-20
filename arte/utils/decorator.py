from functools import wraps
import os
import time
import pickle
import traceback
import inspect
import threading
import types
from astropy.io import fits
from arte.utils.logger import LoggerException
from numpy import ndarray


class ReturnTypeMismatchError(Exception):
    pass


def _raiseReturnTypeMismatchError(expctedType, actualType):
    raise ReturnTypeMismatchError(
        "Type mismatch: "
        "Expected is %s but got %s" % (expctedType, actualType))


def returns(returnType):

    def realDecorator(f):

        @wraps(f)
        def wrapperMethod(*args, **kwds):
            result = f(*args, **kwds)
            if not isinstance(result, returnType):
                resultType = type(result)
                _raiseReturnTypeMismatchError(returnType, resultType)
            return result

        return wrapperMethod

    return realDecorator


def returnsNone(f):
    return returns(type(None))(f)


def returnsForExample(exampleInstance):

    def realDecorator(f):

        @wraps(f)
        def wrapperMethod(*args, **kwds):
            result = f(*args, **kwds)
            resultType = type(result)
            exampleType = type(exampleInstance)
            if resultType != exampleType:
                _raiseReturnTypeMismatchError(exampleType, resultType)
            return result

        return wrapperMethod

    return realDecorator


def suppressException(resultInCaseOfFailure=None):

    def decorate(f):

        @wraps(f)
        def wrapper(self, *args, **kwds):
            try:
                return f(self, *args, **kwds)
            except Exception as e:
                self._logger.error(str(e))
                traceback.print_exc()
                return resultInCaseOfFailure

        return wrapper

    return decorate


def _logEnterAndExit(loggerMethod, enterMessage, exitMessage,
                     f, self, *args, **kwds):
    loggerMethod(enterMessage)
    res = f(self, *args, **kwds)
    loggerMethod(exitMessage)
    return res


def logEnterAndExit(enterMessage, exitMessage, level='notice'):

    def wrapperFunc(f):

        @wraps(f)
        def wrapper(self, *args, **kwds):
            if self._logger is None:
                raise LoggerException(
                    "Logger unavailable for message '%s' '%s'" %
                    (enterMessage, exitMessage))
            loggerMethod = self._logger.__getattribute__(level)
            return _logEnterAndExit(loggerMethod,
                                    enterMessage, exitMessage,
                                    f, self, *args, **kwds)

        return wrapper

    return wrapperFunc


def logTime(f):

    @wraps(f)
    def wrappedMethod(self, *args, **kwds):
        t0 = time.time()
        try:
            return f(self, *args, **kwds)
        finally:
            diffSec = time.time() - t0
            self._logger.notice("Method '%s' took %.3f sec" % (
                f.__name__, diffSec))

    return wrappedMethod


def cacheResult(f):

    @wraps(f)
    def wrapper(self, *args):
        cacheName = f.__name__ + "_cached_result"
        if cacheName not in self.__dict__:
            self.__dict__[cacheName] = {}

        key = ()
        for aa in args:
            if isinstance(aa, ndarray):
                key += (hash(aa.tostring()),)
            else:
                key += (aa,)
        if key not in self.__dict__[cacheName]:
            result = f(self, *args)
            self.__dict__[cacheName][key] = result
        return self.__dict__[cacheName][key]

    return wrapper


def override(f):
    return f


def logFailureAndContinue(func):

    @wraps(func)
    def wrappedMethod(self, *args, **kwds):
        try:
            return func(self, *args, **kwds)
        except Exception as e:
            traceback.print_exc()
            self._logger.error("'%s' failed: %s" % (
                func.__name__, str(e)))

    return wrappedMethod


def _synchronizedWith(lock):

    def decorator(func):

        @wraps(func)
        def synchedFunc(*args, **kwds):
            with lock:
                return func(*args, **kwds)

        return synchedFunc

    return decorator


def _synchronizedWithAttr(lockName):

    def decorator(method):

        @wraps(method)
        def synchronizedMethod(self, *args, **kwds):
            lock = self.__dict__[lockName]
            with lock:
                return method(self, *args, **kwds)

        return synchronizedMethod

    return decorator


def synchronized(item):

    if isinstance(item, str):
        return _synchronizedWithAttr(item)
    elif inspect.isclass(item):
        syncClass = item
        lock = threading.RLock()

        origInit = syncClass.__init__

        def __init__(self, *args, **kwds):
            self.__lock__ = lock
            origInit(self, *args, **kwds)

        syncClass.__init__ = __init__

        for key in syncClass.__dict__:
            val = syncClass.__dict__[key]
            if isinstance(val, types.FunctionType):
                decorator = _synchronizedWith(lock)
                setattr(syncClass, key, decorator(val))

        return syncClass
    else:
        assert False, "Unsupported item type: %s is of type %s" % (
            str(item), type(item))


class FitsFileCache:
    '''Adapter to cache data into FITS files'''

    def check(fname): return os.path.exists(fname)

    def load(fname): return fits.getdata(fname)

    def store(fname, data): fits.writeto(fname, data)


class TextFileCache:
    '''Adapter to cache data into text files'''

    def check(fname): return os.path.exists(fname)

    def load(fname): return open(fname, 'r').read()

    def store(fname, data): open(fname, 'w').write(data)


class PickleFileCache:
    '''Adapter to cache data into files using pickle'''

    def check(fname): return os.path.exists(fname)

    def load(fname): return pickle.load(open(fname, 'r'))

    def store(fname, data): pickle.dump(data, open(fname, 'w'))


def cache_on_disk(fname=None, fname_func=None, adapter=FitsFileCache):
    '''
    Decorator that caches a function result into a local file.

    The file can be specified as either a constant filename, or a function
    that returns a filename. The second form is useful if the filename is not
    known when the function is defined, but only at runtime.

    Roughly equivalent to:

        if adapter.check(fname):
            data = adapter.load(fname)
        else:
            data = f()
            adapter.store(fname, data)
        return data

    Parameters
    ----------
    fname : str, optional
        name of the file where the data is stored. One of `filename` and
        `fname_func` must be specified.
    fname_func : callable, optional
        function or callable without arguments that returns a filename,
        to be used in case the name changes at runtime.
    adapter : class
        class managing the load/store to file, optional. By default FITS
        files will be used as storage.
    '''
    if (fname is None and fname_func is None) or \
       (fname is not None and fname_func is not None):
        raise ValueError('One of filename and fname_func must be specified')

    # Always use fname_func, binding the current value of filename.
    if fname:

        def fname_func():
            return fname

    def decorator(f):

        @wraps(f)
        def cache_result_into_files(*args, **kwargs):
            fname = fname_func()
            if adapter.check(fname):
                return adapter.load(fname)
            else:
                data = f(*args, **kwargs)
                adapter.store(fname, data)
                return data

        return cache_result_into_files

    return decorator

