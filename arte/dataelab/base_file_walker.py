import abc

class AbstractFileNameWalker(metaclass=abc.ABCMeta):
    """Abstract base class for file path management in data analysis.
    
    FileWalker knows the directory structure and file naming conventions
    of your data storage. It provides methods to:
    
    - Locate the root directory of data storage
    - Find the directory for a specific tag
    - Return full paths for each data file type
    - Find tags within a date range
    
    Derived classes must implement snapshot_dir() and typically add
    one method per file type to return complete file paths.
    
    Examples
    --------
    >>> class MyFileWalker(AbstractFileNameWalker):
    ...     def snapshot_dir(self, tag):
    ...         root = Path(os.environ['DATA_ROOT'])
    ...         day = tag.split('_')[0]
    ...         return root / day / tag
    ...     
    ...     def camera_frames(self, tag):
    ...         return self.snapshot_dir(tag) / 'frames.fits'
    ...     
    ...     def dm_commands(self, tag):
    ...         return self.snapshot_dir(tag) / 'dm_commands.npy'
    
    >>> walker = MyFileWalker()
    >>> frames_file = walker.camera_frames('20240101_120000')
    
    See Also
    --------
    BaseAnalyzer : Uses FileWalker to locate data files
    """

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

    def find_tag_between_dates(self, tag_start, tag_stop):
        '''Return the list of tags between the two extremes'''
        return []
