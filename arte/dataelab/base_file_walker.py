import abc

class AbstractFileNameWalker:

    @abc.abstractmethod
    def snapshots_dir(self):
        '''Return the directory where snapshots are stored'''
        pass

    @abc.abstractmethod
    def snapshot_dir(self, tag):
        '''Return the director where a single snapshot is stored'''

    @abc.abstractmethod
    def find_tag_between_dates(self, tag_start, tag_stop):
        '''Return the list of tags between the two extremes'''
        pass
