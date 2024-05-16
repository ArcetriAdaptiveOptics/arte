from arte.utils.timestamp import Timestamp


class Tag(object):

    def __init__(self, tag_string):
        assert tag_string.count("_") > 0
        self._tag_string = tag_string

    def get_day_as_string(self):
        return self._tag_string.split("_")[0]

    def __str__(self):
        return self._tag_string

    @staticmethod
    def create_tag():
        '''Create a tag with the current timestamp'''
        return Tag(Timestamp().now())
