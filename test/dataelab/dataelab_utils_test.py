# -*- coding: utf-8 -*-

import logging
import unittest

from arte.dataelab.dataelab_utils import _is_logging_configured

class LoggingTest(unittest.TestCase):

    def test_is_configured_when_not_configured(self):
        
        root = logging.getLogger()
        handlers = root.handlers.copy()
        for handler in handlers:
            root.removeHandler(handler)
        
        assert _is_logging_configured() == False

    def test_is_configured_when_configured(self):
        
        root = logging.getLogger()
        handlers = root.handlers.copy()
        for handler in handlers:
            root.removeHandler(handler)
        logging.basicConfig()
        
        assert _is_logging_configured() == True
