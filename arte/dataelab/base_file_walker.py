import abc

class AbstractFileNameWalker(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def snapshot_dir(self, tag):
        '''Return the directory with a specific snapshot's contents'''

    def find_tag_between_dates(self, tag_start, tag_stop):
        '''Return the list of tags between the two extremes'''
        return []
