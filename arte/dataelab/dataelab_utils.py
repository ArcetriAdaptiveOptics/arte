'''
Utility functions strictly reserved for dataelab classes
'''
import logging

def setup_dataelab_logging():
    '''
    Setup a default logging style for dataelab classes.
    
    If logging has already been configured, this function does nothing.
    '''
    return # Disabled for now

    if not _is_logging_configured():
        logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            level=logging.WARNING)


def _is_logging_configured():
    '''Detect whether logging has already been configured'''
    root = logging.getLogger()
    return root.hasHandlers()
