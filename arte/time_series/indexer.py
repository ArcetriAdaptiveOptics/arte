import numpy as np


class Indexer(object):

    def __init__(self):
        self.dicCoord = {'x': 0, 'y': 1}

    def _xy_index(self, coord):
        if not hasattr(coord, '__iter__'):
            coord = [coord]
        indexCoord = [self.dicCoord[x] for x in coord]
        return indexCoord

    def _interleaved_xy(self, *args, **kwargs):
        '''
        default: coord
        Accepted keywords:
        coord= x and/or y
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        coord = ['x', 'y']
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if k == 'axis':
                if v == 'x':
                    return slice(0, None, 2)
                elif v == 'y':
                    return slice(1, None, 2)
                else:
                    raise Exception("%s not in accepted"
                                    " values ('x', 'y')" % (v))
        return self._xy_index(coord)

    def xy(self, *args, **kwargs):
        '''
        default: coord
        Accepted keywords:
        coord= x and/or y
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        coord = ['x', 'y']
        if len(args) > 0:
            coord = args[1]
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if k == 'coord' or k == 'coords' or k == 'axis':
                coord = v
        return self._xy_index(coord)

    # TODO add boundary checks with maxrange on args
    @staticmethod
    def myrange(*args, maxrange):
        
        if len(args)==0:
            return range(maxrange)
        if len(args)==1:
            return range(args[0])
        if len(args)==2:
            return range(args[0], args[1])
        if len(args)==3:
            return range(args[0], args[1], args[2])
        raise ValueError
            
class ModeIndexer(Indexer):

    def __init__(self, max_mode=None):
        super().__init__()
        self._max_mode = max_mode        

    def modes(self, *args, **kwargs):
        '''
        default: all modes
        mode = single mode
        modes = list of modes
        from_mode = first mode
        to_mode = last mode
        '''
        modes = None
        from_mode = None
        to_mode = None
        if (len(args) > 0) & (len(args) < 2):
            modes = args[0]
        for k, v in kwargs.items():
            if k == 'modes' or k == 'mode':
                modes = v
            if k == 'from_mode':
                from_mode = v
            if k == 'to_mode':
                to_mode = v
        if modes is not None:
            return modes
        if from_mode is None:
            from_mode = 0
        if to_mode is None:
            to_mode = self._max_mode
        if to_mode is None:
            raise Exception("to_mode was not specified and max_mode was not set "
                            "when initializing the ModeIndexer, cannot continue") 
        return np.arange(from_mode, to_mode)

