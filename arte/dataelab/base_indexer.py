from itertools import chain  # List flattening

class BaseIndexer():

    def __init__(self, single_kw=None, from_kw=None, to_kw=None):
        self.single_kw = ['element', 'elements']
        self.from_kw= ['from_element', 'first']
        self.to_kw = ['to_element', 'last']

        if single_kw:
            self.single_kw += [single_kw]
            self.single_kw = list(chain.from_iterable(self.single_kw))
        if from_kw:
            self.from_kw += [from_kw]
            self.from_kw = list(chain.from_iterable(self.from_kw))
        if to_kw:
            self.to_kw += [to_kw]
            self.to_kw = list(chain.from_iterable(self.to_kw))

    def process_args(self, *args, **kwargs):
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
        if len(args) > 0:
            elements = args
        else:
            for k, v in kwargs.items():
                if k in self.single_kw:
                    elements = v
                if k in self.from_kw:
                    from_element = v
                if k in self.to_kw:
                    to_element = v
        if elements is not None:
            return elements
        return slice(from_element, to_element)