import numpy as np
from numbers import Number


class ModalCoefficients():
    '''
    Coefficients of a modal decomposition, each one labelled by the index
    of its mode.

    Parameters
    ----------
    coefficients: `~numpy.ndarray`
        Coefficient values.
    counter: int, optional
        Counter associated to the coefficients.
    first_mode: int, optional
        Index of the first mode, used when mode_indexes is not given:
        the modes are labelled first_mode, first_mode + 1, ...
        Default 0.
    mode_indexes: sequence of int, optional
        Mode index of each coefficient, in any order and without
        repetitions. It must have the same length as coefficients.
        When given, first_mode is ignored.

    Notes
    -----
    Mode labels are fixed at construction time: FIRST_MODE is read-only.
    Arithmetic operations between two instances act mode by mode on the
    union of their mode indexes, missing modes counting as zero.
    '''

    def __init__(self, coefficients, counter=0, first_mode=0,
                 mode_indexes=None):
        self._coefficients = coefficients
        self._counter = counter
        if mode_indexes is None:
            mode_indexes = np.arange(first_mode,
                                     first_mode + len(coefficients))
        self._mode_indexes = self._checkModeIndexes(
            mode_indexes, len(coefficients))
        self._first_mode = first_mode

    @staticmethod
    def _checkModeIndexes(mode_indexes, n_coefficients):
        idx = np.array(mode_indexes)
        if idx.size == 0:
            idx = idx.astype(int)
        if idx.ndim != 1 or idx.dtype.kind not in 'iu':
            raise ValueError(
                'mode_indexes must be a 1D sequence of integers, got %s' %
                repr(mode_indexes))
        if len(idx) != n_coefficients:
            raise ValueError(
                'mode_indexes has %d elements, coefficients have %d' % (
                    len(idx), n_coefficients))
        if len(np.unique(idx)) != len(idx):
            raise ValueError('mode_indexes must not contain duplicates: %s' %
                             idx)
        idx.setflags(write=False)
        return idx

    @property
    def FIRST_MODE(self):
        '''Index of the first stored mode (read-only)'''
        if len(self._mode_indexes) == 0:
            return self._first_mode
        return int(self._mode_indexes[0])

    @FIRST_MODE.setter
    def FIRST_MODE(self, value):
        raise AttributeError(
            '%s: mode indexes are fixed at construction time; build a new '
            'object with first_mode= or mode_indexes= instead' %
            self.__class__.__name__)

    def modeIndexes(self):
        '''Mode index of each coefficient, in the same order as
        toNumpyArray()'''
        return self._mode_indexes

    def numberOfModes(self):
        return len(self._coefficients)

    def getM(self, modeIndexes):
        '''
        Return the coefficient(s) of the given mode index(es).

        Parameters
        ----------
        modeIndexes: int or array_like of int
            Mode index(es), among those returned by modeIndexes().

        Returns
        -------
        Coefficient(s) with the same shape as modeIndexes.

        Raises
        ------
        IndexError
            If any index is not among modeIndexes(), or if the indexes
            are not integers (booleans included).
        '''
        wanted = np.array(modeIndexes)
        if wanted.dtype.kind not in 'iu':
            raise IndexError(
                '%s: mode indexes must be integers, got %s' % (
                    self.__class__.__name__, repr(modeIndexes)))
        flat = wanted.ravel()
        sorter = np.argsort(self._mode_indexes, kind='stable')
        sorted_idx = self._mode_indexes[sorter]
        pos = np.searchsorted(sorted_idx, flat)
        found = pos < len(sorted_idx)
        found[found] = sorted_idx[pos[found]] == flat[found]
        if not np.all(found):
            raise IndexError(
                '%s: mode index(es) %s not available; available modes: %s' % (
                    self.__class__.__name__, flat[~found], self._mode_indexes))
        return self.toNumpyArray()[sorter[pos].reshape(wanted.shape)]

    def toDictionary(self):
        keys = self.modeIndexes()
        values = self._coefficients
        return dict(list(zip(keys, values)))

    def toNumpyArray(self):
        return self._coefficients

    @staticmethod
    def fromNumpyArray(coefficientsAsNumpyArray, counter=0, **kwargs):
        return ModalCoefficients(np.array(coefficientsAsNumpyArray), counter, **kwargs)

    def counter(self):
        return self._counter

    def setCounter(self, counter):
        self._counter = counter

    def __setstate__(self, state):
        # Objects pickled before mode_indexes existed store FIRST_MODE
        state = dict(state)
        first_mode = state.pop('FIRST_MODE', 0)
        state.setdefault('_first_mode', first_mode)
        if '_mode_indexes' not in state:
            state['_mode_indexes'] = self._checkModeIndexes(
                np.arange(first_mode,
                          first_mode + len(state['_coefficients'])),
                len(state['_coefficients']))
        self.__dict__.update(state)

    def _new(self, coefficients, mode_indexes=None):
        '''
        New instance of the same class holding coefficients, labelled by
        mode_indexes (default: the same labels as self).
        '''
        if mode_indexes is None:
            mode_indexes = self._mode_indexes
        new = self.__class__.__new__(self.__class__)
        ModalCoefficients.__init__(new, coefficients,
                                   mode_indexes=mode_indexes)
        return new

    def _canOperateWith(self, other):
        return isinstance(other, ModalCoefficients)

    def _modeWiseSum(self, other):
        '''
        Sum self and other mode by mode.

        Returns (coefficients, mode_indexes). The result spans the union
        of the two sets of mode indexes, missing modes counting as zero.
        The order is the one of self followed by the modes only in other;
        if both are sorted the result is sorted too.
        '''
        a, ia = self._coefficients, self._mode_indexes
        b, ib = other._coefficients, other._mode_indexes
        dtype = np.result_type(a, b)
        if np.array_equal(ia, ib):
            return np.add(a, b, dtype=dtype), ia
        extra = ib[~np.isin(ib, ia)]
        labels = np.concatenate((ia, extra))
        if np.all(np.diff(ia) > 0) and np.all(np.diff(ib) > 0):
            labels = np.sort(labels)
        c = np.zeros(len(labels), dtype=dtype)
        sorter = np.argsort(labels)
        c[sorter[np.searchsorted(labels, ia, sorter=sorter)]] += a
        c[sorter[np.searchsorted(labels, ib, sorter=sorter)]] += b
        return c, labels

    def __eq__(self, o):
        if self._counter != o._counter:
            return False
        if not np.array_equal(self._mode_indexes, o._mode_indexes):
            return False
        if not np.array_equal(self._coefficients, o._coefficients):
            return False
        return True

    def __ne__(self, o):
        return not self.__eq__(o)

    def __str__(self):
        return str(self._coefficients)

    def __add__(self, other):
        if self._canOperateWith(other):
            return self._new(*self._modeWiseSum(other))
        if isinstance(other, Number):
            return self._new(self._coefficients + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if self._canOperateWith(other):
            self._coefficients, self._mode_indexes = self._modeWiseSum(other)
            return self
        elif isinstance(other, Number):
            self._coefficients += other
            return self
        return NotImplemented

    def __neg__(self):
        return self._new(-self._coefficients)

    def __pos__(self):
        return self._new(self._coefficients)

    def __abs__(self):
        return pow(sum(coo**2 for coo in self._coefficients), 0.5)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        self += -other
        return self

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._new(self._coefficients * other)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if isinstance(other, Number):
            self._coefficients *= other
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._new(self._coefficients / other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return self._new(other / self._coefficients)
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, Number):
            self._coefficients /= other
            return self
        return NotImplemented
