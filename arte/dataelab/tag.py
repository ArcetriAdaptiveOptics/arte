from arte.utils.timestamp import Timestamp


class Tag(object):
    """Unique identifier for a dataset snapshot.
    
    A Tag is a string identifier (typically a timestamp) that uniquely
    identifies a collection of data files representing a snapshot of the
    system state at a specific time. Tags are used to:
    
    - Organize data by acquisition time
    - Group related files (frames, commands, calibrations)
    - Cache computed results
    - Enable batch analysis across multiple acquisitions
    
    Tags must contain at least one underscore and typically follow the
    format: YYYYMMDD_HHMMSS (e.g., '20240101_120000')
    
    Parameters
    ----------
    tag_string : str
        Tag identifier string. Must contain at least one underscore.
    
    Examples
    --------
    >>> tag = Tag('20240101_120000')
    >>> print(tag)  # '20240101_120000'
    >>> day = tag.get_day_as_string()  # '20240101'
    
    >>> # Create tag with current timestamp
    >>> tag = Tag.create_tag()
    
    Notes
    -----
    The day portion is extracted as everything before the first underscore,
    allowing flexible tag formats as long as they start with a date-like
    identifier.
    
    See Also
    --------
    arte.utils.timestamp.Timestamp : For creating timestamps
    BaseAnalyzer : Uses tags to identify datasets
    """

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
