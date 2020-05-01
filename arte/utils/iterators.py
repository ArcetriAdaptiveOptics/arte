# -*- coding: utf-8 -*-

from itertools import tee


def flatten(x):
    '''
    Generator that flatten arbitrarily nested lists.

    This generator will flatten a list that may contain
    other lists (nested arbitrarily) and simple items
    into a flat list.

    >>> flat = flatten([[1,[2,3]],4,[5,6]])
    >>> list(flat)
    [1, 2, 3, 4, 5, 6]
    '''
    for item in x:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def pairwise(iterable):
    '''
    Pairwise iterator, from itertool recipes.

    See Python library docs, itertools chapter.

    >>> s = [0,1,2,3,4,5,6]
    >>> list(pairwise(s))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    '''
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
