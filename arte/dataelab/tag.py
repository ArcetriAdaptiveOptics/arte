

from arte.utils.timestamp import Timestamp


class Tag(object):

    def __init__(self, tagString):
        assert tagString.count("_") > 0
        self._tagString = tagString

    def get_day_as_string(self):
        return self._tagString.split("_")[0]

    def __str__(self):
        return self._tagString

    @staticmethod
    def create_tag():
        return Tag(Timestamp().now())
