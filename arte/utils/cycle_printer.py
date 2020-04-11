
import time


class CyclePrinter():
    '''
    Small class to give information about iterative loops

    Initialize with a name (that will appear as the first word in the
    output message), an optional minimum period between messages
    in seconds (defualt 1.0), and an optional function that will be
    called with a single string argument (defaults to the system
    "print" function), and a format for the number of seconds.

    Call the 'cycle' function at each iteration. If at least "period"
    second have passed from the last call, the "logFunc" function
    will be called with a descriptive message as a string argument.
    '''

    def __init__(self, name, period=1.0, logFunc=print, fmt='%5.2f'):
        self.name = name
        self.period = period
        self.cycleCounter = 0
        self.totalCounter = 0
        self.start = None
        self.prevCycle = None
        self.logFunc = logFunc
        self.fmt = fmt

    def cycle(self):
        self.cycleCounter += 1
        self.totalCounter += 1
        now = time.time()
        if self.start is None:
            self.start = now
            self.prevCycle = now
            return

        if now - self.prevCycle > self.period:
            self.elapsedTime = now-self.prevCycle
            self.logFunc( self._msg())
            self.prevCycle = now
            self.cycleCounter = 0

    def _msg(self):
        msg=('%s: %d cycles in '+self.fmt+' seconds') % \
            (self.name,
             self.cycleCounter,
             self.elapsedTime)
        return msg


class PercentPrinter(CyclePrinter):
    '''
    Small class to give information about iterative loops

    Specializaton of CyclePrinter for processes that go from 0% to 100%.
    Initialize with the same arguments as CyclePrinter and an additional
    "total" argument that represents the number of times the "cycle"
    function will be called. "total" defaults to 100.
    '''

    def __init__(self, name, period=1.0, logFunc=print, fmt='%5.2f', total=100):

        CyclePrinter.__init__(self, name, period, logFunc, fmt)
        self.total = total

    def _msg(self):

        percent = int(self.counter*100.0 / self.total)
        msg=('%s: %d done (%d out of %d) in '+self.fmt+' seconds') % \
            (self.name,
             percent,
             self.counter,
             self.total,
             self.elapsedTime)
        return msg

class SpeedPrinter(CyclePrinter):
    '''
    Small class to give information about iterative loops

    Similar to CyclePrinter, but prints the loop speed
    instead of the number of cycles
    '''

    def __init__(self, name, period=1.0, logFunc=print, fmt='%5.2f'):
        CyclePrinter.__init__(self, name, period, logFunc, fmt)
        self.fmt = fmt

    def _msg(self):

        speed = self.cycleCounter / self.elapsedTime
        msg=('%s: '+self.fmt+' iteration/sec') % (self.name, speed)
        return msg

