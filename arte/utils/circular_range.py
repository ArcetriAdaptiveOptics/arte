

class CircularRange():
    '''Circular range()-like object.'''

    def __init__(self, start, stop, modulo):
        self.start = start
        self.stop = stop
        self.modulo = modulo
        self.index = start

    def __repr__(self):
        return 'CircularRange(%d, %d, %d)' % (self.start, self.stop, self.modulo)

    def __iter__(self):
        while self.index != self.stop:
            yield self.index
            self.index = (self.index + 1) % self.modulo


