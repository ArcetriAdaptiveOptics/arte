
import datetime

__version__= "$Id: timestamp.py 97 2016-04-17 17:14:31Z lbusoni $"


class Timestamp(object):

    def __init__(self):
        self._now= datetime.datetime.now()


    def asNowString(self):
        return self._now.strftime("%Y%m%d_%H%M%S")


    def asTodayString(self):
        return self._now.strftime("%Y%m%d")


    @staticmethod
    def now():
        return Timestamp().asNowString()


    @staticmethod
    def today():
        return Timestamp().asTodayString()


    @staticmethod
    def nowUSec():
        ss= datetime.datetime.now()
        return ss.strftime('%Y%m%d_%H%M%S')


    def __str__(self):
        return self.asNowString()
