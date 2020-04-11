
import time

class Timer(object):
    '''
    Timer context manager. Originally from
    https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

    and modified to support multiple time units

    Example:

    with Timer(unit='s', msg='foo', precision=2):
        long_function()

    result:
    foo: Elapsed: 33.20 s
    '''
    def __init__(self, unit='s', precision=2, msg=''):
        self.unit = unit
        self.precision = precision
        self.msg = msg
        if unit not in ['s', 'ms', 'us']:
            raise Exception('Unit %s is not supported' % unit)

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = time.time() - self.t0

        name = self.name
        if name != '':
            name = ' (%s)' % name

        fmt = 'Elapsed time (%s): %%.%df %s' % (self.msg, self.precision, self.unit)

        number = {'s': elapsed,
                  'ms': elapsed * 1000,
                  'us': elapsed * 1000000 }

        print(fmt % number)


