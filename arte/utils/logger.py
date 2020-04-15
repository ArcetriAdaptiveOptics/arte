import abc
import os
import inspect


class LoggerException(Exception):
    pass


class AbstractLogger(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fatal(self, m):
        raise RuntimeError("Do not use me!")

    @abc.abstractmethod
    def error(self, m):
        raise RuntimeError("Do not use me!")

    @abc.abstractmethod
    def warn(self, m):
        raise RuntimeError("Do not use me!")

    @abc.abstractmethod
    def notice(self, m):
        raise RuntimeError("Do not use me!")

    @abc.abstractmethod
    def debug(self, m):
        raise RuntimeError("Do not use me!")


class PythonLogger(AbstractLogger):

    def __init__(self, logger):
        self._logger= logger
        self._radix= ''


    def setRelativePathRadix(self, radix):
        self._radix= radix


    def _getRelativeDirName(self, fullPath):
        return os.path.relpath(fullPath, self._radix)


    def _getCallerTrace(self):
        stackCaller= inspect.stack()[3]
        return '%s:%s' % (self._getRelativeDirName(stackCaller[1]),
                          stackCaller[2])

    def _formatMessage(self, m):
        return '%s %s: %s' % (self._logger.name, self._getCallerTrace(), m)

    def fatal(self, m):
        msg= self._formatMessage(m)
        self._logger.fatal(msg)

    def error(self, m):
        msg= self._formatMessage(m)
        self._logger.error(msg)

    def warn(self, m):
        msg= self._formatMessage(m)
        self._logger.warning(msg)

    def notice(self, m):
        msg= self._formatMessage(m)
        self._logger.info(msg)

    def debug(self, m):
        msg= self._formatMessage(m)
        self._logger.debug(msg)


class ObservableLogger(AbstractLogger):


    def __init__(self, wrappedLogger):
        self._wrappedLogger= wrappedLogger
        self._listeners= []

    def fatal(self, m):
        for each in self._listeners:
            each.onFatal(m)
        self._wrappedLogger.fatal(m)


    def error(self, m):
        for each in self._listeners:
            each.onError(m)
        self._wrappedLogger.error(m)


    def warn(self, m):
        for each in self._listeners:
            each.onWarning(m)
        self._wrappedLogger.error(m)


    def notice(self, m):
        for each in self._listeners:
            each.onNotice(m)
        self._wrappedLogger.notice(m)


    def debug(self, m):
        for each in self._listeners:
            each.onDebug(m)
        self._wrappedLogger.debug(m)


    def addListener(self, listener):
        self._listeners.append(listener)


class LoggerListener(object):


    def onDebug(self, message):
        self._methodOfPureAbstractClass()


    def onNotice(self, message):
        self._methodOfPureAbstractClass()


    def onWarning(self, message):
        self._methodOfPureAbstractClass()


    def onError(self, message):
        self._methodOfPureAbstractClass()


    def onFatal(self, message):
        self._methodOfPureAbstractClass()


    def _methodOfPureAbstractClass(self):
        assert False, "Pure abstract class method"


class DummyLogger(AbstractLogger):

    def fatal(self, m):
        pass

    def error(self, m):
        pass

    def warn(self, m):
        pass

    def notice(self, m):
        pass

    def debug(self, m):
        pass


class AbstractLoggerFactory(object):

    def getLogger(self, loggerName):
        raise RuntimeError("Do not use me!")


class PythonLoggerFactory(AbstractLoggerFactory):

    def getLogger(self, name):
        import logging

        logger= logging.getLogger(name)
        pyLogger= PythonLogger(logger)
        radix= os.path.join(__file__, '..', '..')
        pyLogger.setRelativePathRadix(radix)
        return pyLogger


class DummyLoggerFactory(AbstractLoggerFactory):

    def getLogger(self, name):
        return DummyLogger()


class ObservableLoggerFactory(AbstractLoggerFactory):

    def __init__(self, wrappedLoggerFactory):
        self._wrappedLoggerFactory= wrappedLoggerFactory
        self._createdLoggerByName= {}
        self._listeners= []


    def _registerListenersTo(self, logger):
        for each in self._listeners:
            logger.addListener(each)


    def getLogger(self, name):
        if name in list(self._createdLoggerByName.keys()):
            return self._createdLoggerByName[name]

        wrappedLogger= self._wrappedLoggerFactory.getLogger(name)
        logger= ObservableLogger(wrappedLogger)
        self._createdLoggerByName[name]= logger

        self._registerListenersTo(logger)

        return logger


    def getLoggerMap(self):
        return self._createdLoggerByName


    def addListener(self, loggerListener):
        self._listeners.append(loggerListener)
        for loggerName in self._createdLoggerByName:
            logger= self._createdLoggerByName[loggerName]
            logger.addListener(loggerListener)


class Logger(object):

    DEFAULT_LOGGER_FACTORY= PythonLoggerFactory
    _loggerFactory= DEFAULT_LOGGER_FACTORY()


    @staticmethod
    def of(loggerName):
        return Logger._loggerFactory.getLogger(loggerName)


    @staticmethod
    def setLoggerFactory(loggerFactory):
        Logger._loggerFactory= loggerFactory
