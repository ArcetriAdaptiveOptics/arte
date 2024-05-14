from datetime import datetime


class Tag(object):
    '''A *tag* is an unique snapshot identifier in the format SYSTEM_YYYYMMDD_HHMMSS
    
    where SYSTEM is an user-defined string for system under analysis, and the rest
    is a timestamp.

    When saving a new snapshot, do not use the standard constructor. Use the
    create_tag() classmethod instead.
    
    parameters
    ---------
    tag: str
        snapshot tag
    '''
    def __init__(self, tag):
        assert tag.count("_") == 2
        self._tag = tag

    def get_system(self):
        '''Return system prefix as a string'''
        return self._tag.split('_')[0]

    def get_day_string(self):
        '''Return day string in YYYYMMDD format'''
        return self._tag.split('_')[1]

    def get_time_string(self):
        '''Return time string in HHHMMSS format'''
        return self._tag.split('_')[2]

    def get_year(self):
        '''Return year as an integer number'''
        return int(self.get_day_string()[:4])

    def get_month(self):
        '''Return month as an integer number'''
        return int(self.get_day_string()[4:6])

    def get_day(self):
        '''Return day as an integer number'''
        return int(self.get_day_string()[6:8])

    def get_hour(self):
        '''Return hour as an integer number'''
        return int(self.get_time_string()[:2])

    def get_minute(self):
        '''Return minute as an integer number'''
        return int(self.get_time_string()[2:4])

    def get_second(self):
        '''Return second as an integer number'''
        return int(self.get_time_string()[4:6])

    def __str__(self):
        return self._tag

    def get_datetime(self):
        '''Return a datetime object corresponding to this tag'''
        day = self.get_day_string()
        time = self.get_time_string()
        return datetime.strptime(f'{day} {time}',
                                 '%Y%m%d %H%M%S')

    @staticmethod
    def create_tag(system):
        '''Create a tag with a system prefix and the current timestamp
        
        Parameters
        ----------
        system: str
            system identifier string
            
        Returns
        -------
        tag: str
            complete tag string with system and the current timestamp
        '''
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Tag(f'{system}_{now}')
