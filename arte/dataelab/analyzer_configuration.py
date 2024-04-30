

from enum import Enum


class ConfItem(Enum):
    SNAPSHOT_DIR = 1
    TMP_DIR = 2


class BaseConfiguration():
    '''
    Key-based configuration with two mandatory items:
        - ConfItem.SNAPSHOT_DIR
        - ConfItem.TMP_DIR
        
    Other user-defined items can be added with add()
    '''
    def __init__(self, snapshot_dir, tmp_dir, **kwargs):
        self._data = kwargs.copy()
        self._data[ConfItem.SNAPSHOT_DIR] = snapshot_dir
        self._data[ConfItem.TMP_DIR] = tmp_dir

    def snapshot_dir(self):
        return self._data[ConfItem.SNAPSHOT_DIR]
    
    def tmp_dir(self):
        return self._data[ConfItem.TMP_DIR]
    
    def add(self, key, value):
        self._data[key] = value

    def item(self, key):
        return self._data[key]