'''
Utility functions strictly reserved for dataelab classes
'''
def is_dataelab(x):
    '''Return whether x is supports the dataelab protocol'''
    return hasattr(x, 'get_data') and hasattr(x, 'astropy_unit')
