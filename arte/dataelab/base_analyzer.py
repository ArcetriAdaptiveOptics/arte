import datetime
import functools

from arte.utils.help import add_help
from arte.dataelab.cache_on_disk import set_tag, clear_cache


class PostInitCaller(type):
    '''Meta class with a _post_init() method.
    
    Used to initialize the disk cache, since DiskCacher objects are
    available only after all child/member objects have been initialized'''
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._post_init()
        return obj


@add_help
class BaseAnalyzer(metaclass=PostInitCaller):
    '''Main analyzer object
    
    Note that an analyzer is not CanBeIncomplete, because all attributes
    should be initialized (if they are missing, their value will be NA)
    '''

    def __init__(self, snapshot_tag, recalc=False):
        self._snapshot_tag = snapshot_tag
        self._recalc = recalc
 
    def _post_init(self):
        '''Initialize disk cache
        
        Executed after all child classes have completed their __init__
        thanks to the PostInitCaller metaclass
        '''
        set_tag(self, self._snapshot_tag)
        if self._recalc:
            self.recalc()
            self._recalc = False

    @classmethod
    def get(cls, tag, *args, recalc=False, **kwargs):
        '''Get the Analyzer instance (or derived class) corresponding to *tag*.

        This method mantains an internal cache. If a tag is requested
        multiple times, the same Analyzer instance is returned.
        
        Parameters
        ----------
        tag: str
            snapshot tag
        recalc: bool, optional
            if set to True, any cached data for this tag will be deleted and
            computed again when requested
         '''
        analyzer = cls._get(tag, *args, **kwargs)
        # Special recalc handling, must be set again for a cached instance
        if recalc:
            analyzer.recalc()
        return analyzer

    @classmethod
    @functools.cache
    def _get(cls, tag, *args, recalc=False, **kwargs):
        '''Get a new Analyzer instance 

        Added one level of indirection (get() calls _get())
        to make sure that the cache always works even when
        the default argument *recalc* is not specified in get().

        Also makes it easier to override it in classes
        '''
        return cls(tag, *args, recalc=recalc, **kwargs)

    def recalc(self):
        '''Force recalculation of this analyzer data'''
        clear_cache(self)

    def snapshot_tag(self):
        '''Snapshot tag for this Analyzer object'''
        return self._snapshot_tag

    def date_in_seconds(self):
        '''Tag date as seconds since the epoch'''
        epoch = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
        this_date = datetime.datetime(int(self._snapshot_tag[0:4]),
                                     int(self._snapshot_tag[4:6]),
                                     int(self._snapshot_tag[6:8]),
                                     int(self._snapshot_tag[9:11]),
                                     int(self._snapshot_tag[11:13]),
                                     int(self._snapshot_tag[13:15]))
        td = this_date - epoch
        return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6

    # Override to add additional info
    def _info(self):
        return {'snapshot_tag': self._snapshot_tag}

    def info(self):
        '''Info dictionary'''
        return self._info()

    def summary(self, keywidth=None):
        '''Print info dictionary on stdout'''
        info = self._info()
        if keywidth is None:
            keywidth = max(len(x) for x in info.keys())
        for k, v in info.items():
            print(f'{k:{keywidth}} : {v}')

    def wiki(self, header=True):
        '''Print info dictionary in wiki format on stdout'''
        info = self._info()
        spacing = [max([len(x)+2, len(info[x])]) for x in info.keys()]

        if header:
            for i, k in enumerate(info.keys()):
                print(f'|*{k:^{spacing[i]}}*', end='')
            print('|')

        for i, v in enumerate(info.values()):
            print(f'|{v:^{spacing[i]}}', end='')
        print('|')

