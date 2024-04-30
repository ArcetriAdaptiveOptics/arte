import abc
import datetime
import logging
from arte.utils.not_available import CanBeIncomplete
from arte.utils.help import add_help
from arte.dataelab.cache_on_disk import set_tag


@add_help
class BaseAnalyzer(CanBeIncomplete):
    '''Main analyzer object'''

    def __init__(self, snapshot_tag, recalc=False):
        self._logger = logging.getLogger('an_%s' % snapshot_tag)
        self._snapshot_tag = snapshot_tag
        
        self._logger.info(f'creating analyzer for tag {snapshot_tag}')

        # Initialize tag
        set_tag(self, self._snapshot_tag)

    def snapshot_tag(self):
        return self._snapshot_tag

    def date_in_seconds(self):
        epoch = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
        thisDate = datetime.datetime(int(self._snapshot_tag[0:4]),
                                     int(self._snapshot_tag[4:6]),
                                     int(self._snapshot_tag[6:8]),
                                     int(self._snapshot_tag[9:11]),
                                     int(self._snapshot_tag[11:13]),
                                     int(self._snapshot_tag[13:15]))
        td = thisDate - epoch
        return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6

    def eval(self, commandList):
        resDict = {}
        for cmd in commandList:
            resDict[cmd] = eval('self.' + cmd + '()')
        return resDict

    @abc.abstractmethod
    def _info(self):
        return {}

    def info(self):
        return self._info()

    def summary(self):
        info = self._info()
        spacing = max([len(x) for x in info.keys()])
        fmt = '%%%ds' % spacing
        for k, v in info.items():
            print(fmt % k, ':'+str(v))

    def wiki(self, header=True):
        info = self._info()
        spacing = [max([len(x)+2, len(info[x])]) for x in info.keys()]

        if header:
            for i, k in enumerate(info.keys()):
                bold_k = '*{}*'.format(k)
                fmt = '|{{:^{}}}'.format(spacing[i])
                print(fmt.format(bold_k), end='')
            print('|')

        for i, v in enumerate(info.values()):
            fmt = '|{{:^{}}}'.format(spacing[i])
            print(fmt.format(v), end='')
        print('|')

