import datetime


class Timestamp(object):

    def __init__(self):
        self._now = datetime.datetime.now()

    def now_string(self):
        return self._now.strftime("%Y%m%d_%H%M%S")

    def today_string(self):
        return self._now.strftime("%Y%m%d")

    def timestamp(self):
        return self._now

    @staticmethod
    def now():
        return Timestamp().now_string()

    @staticmethod
    def today():
        return Timestamp().today_string()

    @staticmethod
    def now_usec():
        ss = datetime.datetime.now()
        return ss.strftime('%Y%m%d_%H%M%S.%f')

    def __str__(self):
        return self.now_string()
