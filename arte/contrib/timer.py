
import time

class Timer(object):
    '''
    Timer context manager. Originally from
    https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    with some small embellishments

    Example:

    with Timer('foo', fmt='5.2%f'):
        long_function()

    result:
    foo: Elapsed: 33.20 
    '''
    def __init__(self, name=None, fmt=None):
        self.name = name
        self.fmt = fmt

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = (time.time() - self.tstart)

        if self.name is not None:
            print '%s:' % self.name,

        if self.fmt is None:
            fmt = 'Elapsed: %.3f'
        else:
            fmt = self.fmt
        print fmt % (time.time() - self.tstart)


