import numpy as np


class Indexer(object):

    def __init__(self):
        self.dicCoord = {'x': 0, 'y': 1}

    def _xy_index(self, coord):
        if not hasattr(coord, '__iter__'):
            coord = [coord]
        indexCoord = [self.dicCoord[x] for x in coord]
        return indexCoord

    def interleaved_xy(self, *args, **kwargs):
        '''
        default: coord
        Accepted keywords:
        coord= x and/or y
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if v == 'x':
                return slice(0, None, 2)
            elif v == 'y':
                return slice(1, None, 2)
            else:
                raise Exception("%s not in accepted"
                                " values ('x', 'y')" % (v))

        # Since we are here, no keywords were specified
        if len(args) == 0:
            return slice(0, None, 1)
        elif len(args) == 1:
            if args[0] == 'x':
                return slice(0, None, 2)
            elif args[0] == 'y':
                return slice(1, None, 2)
            elif isinstance(args[0], int):
                return slice(args[0], args[0]+1)
        else:
            raise Exception("%s not in accepted"
                            " values ('x', 'y')" % str(v))

    def sequential_xy(self, maxindex, *args, **kwargs):
        '''
        default: coord
        Accepted keywords:
        coord= x and/or y
        '''
        acceptedKeywords = ['coord', 'coords', 'axis']
        for k, v in kwargs.items():
            if k not in acceptedKeywords:
                raise Exception('possible keywords are ' +
                                str(acceptedKeywords))
            if v == 'x':
                return slice(0, maxindex//2)
            elif v == 'y':
                return slice(maxindex//2, None)
            else:
                raise Exception("%s not in accepted"
                                " values ('x', 'y')" % (v))

        # Since we are here, no keywords were specified
        if len(args) == 0:
            return slice(0, None, 1)
        elif len(args) == 1:
            if args[0] == 'x':
                return slice(0, maxindex//2)
            elif args[0] == 'y':
                return slice(maxindex//2, None)
            elif isinstance(args[0], int):
                return slice(args[0], args[0]+1)
        else:
            raise Exception("%s not in accepted"
                            " values ('x', 'y')" % str(v))
        
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


class DefaultIndexer(Indexer):

    def __init__(self):
        super().__init__()

    def elements(self, *args, **kwargs):
        '''
        default: all elements
        element = single element
        elements = list of elements
        from_element = first element
        to_element = last element
        '''
        elements = None
        from_element = None
        to_element = None
        if (len(args) > 0) & (len(args) < 2):
            elements = args[0]
        for k, v in kwargs.items():
            if k == 'elements' or k == 'element':
                elements = v
            if k == 'from_element':
                from_element = v
            if k == 'to_element':
                to_element = v
        if elements is not None:
            return elements
        return slice(from_element, to_element)


class RowColIndexer(Indexer):

    def __init__(self):
        super().__init__()

    def rowcol(self, *args, **kwargs):
        '''
        rows = index
        cols = index
        '''
        rows = slice(None, None, None)
        cols = slice(None, None, None)
        if len(args) == 1:
            rows = args[0]
        elif len(args) == 2:
            rows, cols = args
        elif len(args) > 2:
            raise IndexError('Unsupported rowcol index')
        
        accepted = 'rows cols row_from col_from row_to col_to row_step col_step'.split()
        for kw in kwargs:
            if kw not in accepted:
                raise IndexError(f'Unsupported index keyword {kw}')

            if kw == 'rows':
                rows = kwargs['rows']
            if kw == 'cols':
                cols = kwargs['cols']
            if kw == 'row_from':
                rows = slice(kwargs['row_from'], rows.stop, rows.step)
            if kw == 'col_from':
                cols = slice(kwargs['col_from'], cols.stop, cols.step)
            if kw == 'row_to':
                rows = slice(rows.start, kwargs['row_to'], rows.step)
            if kw == 'col_to':
                cols = slice(cols.start, kwargs['col_to'], cols.step)
            if kw == 'row_step':
                rows = slice(rows.start, rows.stop, kwargs['row_step'])
            if kw == 'col_step':
                cols = slice(cols.start, cols.stop, kwargs['col_step'])

        return (rows, cols)
