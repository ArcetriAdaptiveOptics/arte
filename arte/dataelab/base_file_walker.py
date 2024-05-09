import abc

class AbstractFileNameWalker:

    @abc.abstractmethod
    def snapshots_dir(self):
        '''Return the directory where snapshots are stored'''
        pass

    @abc.abstractmethod
    def snapshot_dir(self, tag):
        '''
        Abstract method that must be reimplemented in the derived class.

        Parameters
        ----------
        tag: str
            Unique tag that identifies an analyzer data directory

        Returns
        -------
        fullpath: str
            Full path including the tag directory
        '''
        pass

    @abc.abstractmethod
    def find_tag_between_dates(self, tag_start, tag_stop):
        '''Return the list of tags between the two extremes'''
        pass
