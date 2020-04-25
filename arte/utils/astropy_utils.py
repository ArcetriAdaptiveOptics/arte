# -*- coding: utf-8 -*-

import astropy.units as u 

def get_the_unit_if_it_has_one(var):
    '''Returns the argument unit, or 1 if it does not have it'''    
    if isinstance(var, u.Quantity): 
        return var.unit
    else:
        return 1
                 
def match_and_remove_unit(var, var_to_match):
    '''Make sure that var is in the same unit as var_to_match,
       and then remove the unit'''
    if isinstance(var, u.Quantity) and isinstance(var_to_match, u.Quantity):
        # Catch the errors here to have nicer tracebacks, otherwise
        # we end up deep into astropy libraries.
        try:
            return var.to(var_to_match.unit).value
        except Exception as e:
            raise u.UnitsError(str(e))

    elif isinstance(var, u.Quantity):
        return var.value
    else:
        return var

def match_and_remove_units(*args):
    '''
    Take a list of variables, make sure that all have the same unit as the
    first one, and return a list of unitless values plus the common unit.

    Order is significant! The first variable's unit is applied to all others.
    If the first one does not have a unit, all units are removed.
    '''
    if len(args)==0:
        return [1]
    else:
        unit = get_the_unit_if_it_has_one(args[0])

    newvars= [match_and_remove_unit(x, args[0]) for x in args[0:]]
    
    print(args)
    print(newvars)
    
    return newvars+[unit]