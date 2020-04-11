
import itertools

def chunk_iterator(n, iterable):
    '''
    From: https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
    - without permission

    Splits an iterable in chunks of N elements each (but for the last one,
    which might be shorter if needed). Returns an iterator for each chunk.
    Example:
    >>> for a in chunk_iterator(3,range(10)):
    ...  print(list(a))
    ... 
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9]
    '''
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)


