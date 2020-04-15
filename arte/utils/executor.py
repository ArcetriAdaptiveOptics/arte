
from abc import abstractmethod
from functools import wraps
import abc
import threading
import traceback

from concurrent import futures

from arte.utils.decorator import override, synchronized
from arte.utils.logger import Logger


class Task(object, metaclass=abc.ABCMeta):

    @abstractmethod
    def perform(self):
        assert False, "abstract base clase"


    def name(self):
        return "unspecified"


    def __repr__(self):
        return "Task(%r)" % self.name()


class FunctionTaskException(Exception):
    pass


class FunctionTask(Task):

    NO_RESULT= object()

    def __init__(self, func, *args, **kwds):
        self._func= func
        self._args= args
        self._kwds= kwds

        self._result= self.NO_RESULT
        self._exception= None
        self._name= None


    @override
    def perform(self):
        try:
            self._result= self._func(*self._args, **self._kwds)
        except Exception as e:
            self._exception= e
            raise


    @override
    def name(self):
        if self._name:
            return self._name

        return "%s" % self._func


    def __repr__(self):
        return "FunctionTask(%r)" % self.name()


    def withName(self, name):
        self._name= name
        return self


    def getResult(self):
        if self._result is self.NO_RESULT:
            raise FunctionTaskException(
                "No result available for %s" % self.name())
        return self._result


    def getException(self):
        return self._exception


    def reraiseExceptionIfAny(self):
        if self._exception:
            raise self._exception


class TaskControl(object, metaclass=abc.ABCMeta):
    @abstractmethod
    def isDone(self):
        assert False, "abstract base class"


    @abstractmethod
    def isRunning(self):
        assert False, "abstract base class"


    @abstractmethod
    def waitForCompletion(self, timeoutSec):
        assert False, "abstract base class"



class TaskTimeoutError(Exception):

    def __init__(self, msg):
        Exception.__init__(self, msg)


class FutureTaskControl(TaskControl):

    def __init__(self, future, taskName, logger):
        TaskControl.__init__(self)
        assert taskName is not None
        self._future= future
        self._taskName= taskName
        self._logger= logger


    @override
    def isDone(self):
        return self._future.done()


    @override
    def isRunning(self):
        return self._future.running()


    def _printStackTraceInCaseOfError(self):
        try:
            self._future.result()
        except Exception as e:
            self._logger.error("Task '%s' failed: %s" % (
                               self._taskName, str(e)))
            traceback.print_exc()
            self._logger.debug("Details to %s: %s" % (
                               self._taskName, traceback.format_exc()))


    @override
    def waitForCompletion(self, timeoutSec=None):
        doneFutures, _= futures.wait([self._future], timeout=timeoutSec)
        if self._future in doneFutures:
            self._printStackTraceInCaseOfError()
        else:
            raise TaskTimeoutError("Wait for completion"
                                   " of task '%s' failed" % self._taskName)


class Executor(object, metaclass=abc.ABCMeta):

    @abstractmethod
    def execute(self, task):
        assert False, "abstract base class"


class DoneTaskControl(TaskControl):

    @override
    def isDone(self):
        return True


    @override
    def isRunning(self):
        return False


    @override
    def waitForCompletion(self, timeoutSec):
            pass


class SerialExecutor(Executor):

    def __init__(self, logger=Logger.of("SerialExecutor")):
        Executor.__init__(self)
        self._logger= logger


    @override
    def execute(self, task):
        assert task is not None
        self._logger.debug("Executing task '%s'" % task.name())
        try:
            task.perform()
        except Exception as e:
            self._logger.error(str(e))

        return DoneTaskControl()


class MultiThreadExecutor(Executor):

    def __init__(self, maxThreads, logger=Logger.of("MultiThreadExecutor")):
        Executor.__init__(self)
        self._logger= logger
        self._executor= futures.ThreadPoolExecutor(max_workers=maxThreads)


    @override
    def execute(self, task):
        assert task is not None
        self._logger.debug("Executing task '%s'" % task.name())
        future= self._executor.submit(task.perform)
        return FutureTaskControl(future, task.name(), self._logger)


class SingleThreadExecutor(MultiThreadExecutor):

    def __init__(self, logger=Logger.of("SingleThreadExecutor")):
        MultiThreadExecutor.__init__(self, maxThreads=1, logger=logger)


class BusyExecutorException(Exception):

    def __init__(self, message, nameOfCurrentTask):
        Exception.__init__(self, message)
        self.nameOfCurrentTask= nameOfCurrentTask


class SingleThreadImmediateExecutor(Executor):

    def __init__(self, singleThreadExecutor):
        self._executor= singleThreadExecutor
        self._idle= True
        self._mutex= threading.RLock()
        self._currentTask= None


    @override
    @synchronized("_mutex")
    def execute(self, task):
        if not self._idle:
            taskName= self._currentTask.name()
            raise BusyExecutorException(
                "Executor is busy by task '%s'" % (taskName), taskName)

        class TrackedTask(Task):
            def __init__(self, task, onCompletionCallbackFunction):
                Task.__init__(self)
                self._task= task
                self._onCompletionCallback= onCompletionCallbackFunction

            @override
            def perform(self):
                try:
                    self._task.perform()
                finally:
                    self._onCompletionCallback()

            @override
            def name(self):
                return self._task.name()

        self._idle= False
        self._currentTask= task
        return self._executor.execute(TrackedTask(task, self._onCompletion))


    @synchronized("_mutex")
    def _onCompletion(self):
        self._currentTask= None
        self._idle= True


    @synchronized("_mutex")
    def isIdle(self):
        return self._idle


class DelayedTaskControl(TaskControl):


    def __init__(self):
        super(DelayedTaskControl, self).__init__()
        self._done= False


    @override
    def isDone(self):
        return self._done


    def markAsDone(self):
        self._done= True


    @override
    def isRunning(self):
        return False


    @override
    def waitForCompletion(self, timeoutSec):
        assert False, "N/A"


class TaskDelayingExecutor(Executor):


    def __init__(self):
        super(TaskDelayingExecutor, self).__init__()
        self._delayedTasks= []

    @override
    def execute(self, task):
        class WrappedTask(Task):
            def __init__(self, task, delayedTaskControl):
                super(WrappedTask, self).__init__()
                self._task= task
                self._taskControl= delayedTaskControl

            def perform(self):
                self._task.perform()
                self._taskControl.markAsDone()


        taskCtrl= DelayedTaskControl()
        self._delayedTasks.append(WrappedTask(task, taskCtrl))
        return taskCtrl


    def getDelayedTasks(self):
        return self._delayedTasks


    def executeDelayedTasks(self):
        for each in self._delayedTasks:
            each.perform()


class ExecutorFactory(object):


    @staticmethod
    def buildMultiThreadExecutor(maxNumThreads):
        return MultiThreadExecutor(maxNumThreads)


    @staticmethod
    def buildSingleThreadExecutor(logger=Logger.of("SingleThreadExecutor")):
        return SingleThreadExecutor(logger)


    @staticmethod
    def buildSingleThreadImmediateExecutor(
            logger=Logger.of("SingleThreadImmediateExecutor")):
        return SingleThreadImmediateExecutor(
            ExecutorFactory.buildSingleThreadExecutor(logger))



def logTaskFailure(func):
    @wraps(func)
    def wrappedMethod(self, *args, **kwds):
        try:
            return func(self, *args, **kwds)
        except Exception as e:
            message= "Task failed: %s" % str(e)
            self._logger.error(message)
            raise

    return wrappedMethod
