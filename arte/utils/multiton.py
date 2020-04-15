# -*- coding: utf-8 -*-
#########################################################
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-28  Created
#
#########################################################

def multiton(cls):
    '''
    Decorator that returns the same instance of a class
    every time it is instantiated with the same parameters.

    All parameters must be able to be passed to str() in order
    to build an hashable key.
    As a side effect, the class name becomes a function
    that returns an instance, rather than a class type instance.
    '''
    instances = {}
    def getinstance(*args):
        key = str(*args)
        if key not in instances:
            instances[key] = cls(*args)
        return instances[key]
    return getinstance

def multiton_id(cls):
    '''
    Decorator that returns the same instance of a class
    every time it is instantiated with the same parameters.

    Similar to "multiton", but uses the id of each argument
    to build an hashable key. This allows to pass things
    like dictionaries that will be recognized as identical even
    if their contents change, but risks not recognizing identical
    values of strings and numbers.
    As a side effect, the class name becomes a function
    that returns an instance, rather than a class type instance.
    '''
    instances = {}
    def getinstance(*args):
        ids = [str(id(x)) for x in args]
        key = ''.join(ids)
        if key not in instances:
            instances[key] = cls(*args)
        return instances[key]
    return getinstance
