def logFailureAndReturnNotAvailable(func):

    def wrappedMethod(self, *args, **kwds):
        try:
            return func(self, *args, **kwds)
        except Exception as e:
            self._logger.warn("'%s' failed: %s" % (
                func.__name__, str(e)))
            return NotAvailable()

    return wrappedMethod


class CanBeIncomplete(object):
    _verbose = False

    def __getattr__(self, attr):
        if self._verbose:
            print('%s %s __getattr__ %s' % (self.__class__.__name__,
                                            hex(id(self)), attr))
        iPythonSpecials = ['trait_names',
                           '_getAttributeNames',
                           '__length_hint__']
        if attr in iPythonSpecials:
            if self._verbose:
                print('iPython completer I got you!')
            raise AttributeError
        return NotAvailable()


class NotAvailable(CanBeIncomplete):
    _verbose = False

    def __init__(self):
        if self._verbose:
            print('%s __init__ %s' % (hex(id(self)),
                                      self.__class__.__name__))

    def __getitem__(self, key):
        if self._verbose:
            print('%s __getitem__ %s' % (hex(id(self)), str(key)))
        return self

    def __setitem__(self, key, value):
        if self._verbose:
            print('__setitem__ %s %s' % (str(key), str(value)))
        pass

    def __call__(self, *args, **kwargs):
        if self._verbose:
            print('%s __call__ %s %s' % (hex(id(self)),
                                         str(args), str(kwargs)))
        return self

    def __iter__(self):
        return self

    def next(self):
        raise StopIteration

    def __repr__(self):
        return 'NA'

    def __len__(self):
        return 0

    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __floordiv__(self, other):
        return self

    def __mod__(self, other):
        return self

    def __divmod__(self, other):
        return self

    def __pow__(self, other):
        return self

    def __lshift__(self, other):
        return self

    def __rshift__(self, other):
        return self

    def __and__(self, other):
        return self

    def __xor__(self, other):
        return self

    def __or__(self, other):
        return self

    def __div__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __rdiv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __rfloordiv__(self, other):
        return self

    def __rmod__(self, other):
        return self

    def __rdivmod__(self, other):
        return self

    def __rpow__(self, other):
        return self

    def __rlshift__(self, other):
        return self

    def __rrshift__(self, other):
        return self

    def __rand__(self, other):
        return self

    def __rxor__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __iadd__(self, other):
        return self

    def __isub__(self, other):
        return self

    def __imul__(self, other):
        return self

    def __idiv__(self, other):
        return self

    def __itruediv__(self, other):
        return self

    def __ifloordiv__(self, other):
        return self

    def __imod__(self, other):
        return self

    def __ipow__(self, other):
        return self

    def __ilshift__(self, other):
        return self

    def __irshift__(self, other):
        return self

    def __iand__(self, other):
        return self

    def __ixor__(self, other):
        return self

    def __ior__(self, other):
        return self

    def __neg__(self):
        return self

    def __pos__(self):
        return self

    def __abs__(self):
        return self

    def __invert__(self):
        return self

    @staticmethod
    def transformInNotAvailable(obj):
        obj.__class__ = NotAvailable

    @staticmethod
    def isNotAvailable(obj):
        return isinstance(obj, NotAvailable)
