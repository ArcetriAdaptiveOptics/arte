#!/usr/bin/env python


import logging
import random
import threading
import time
import unittest

from apposto.utils.decorator import override, synchronized
from apposto.utils.executor import Task, SingleThreadExecutor, Executor, \
    SingleThreadImmediateExecutor, BusyExecutorException, FunctionTask,\
    MultiThreadExecutor, TaskTimeoutError, DoneTaskControl, logTaskFailure,\
    TaskDelayingExecutor, FunctionTaskException
from apposto.utils.logger import ObservableLogger, DummyLogger,\
    LoggerListener
from test.test_helper import Poller, Probe, TestHelper, ExecutionProbe


__version__= "$Id: executor_test.py 167 2016-12-06 21:31:22Z lbusoni $"


class TaskCompletionProbe(Probe):

    def __init__(self, task):
        super(TaskCompletionProbe, self).__init__()
        self._task= task

    def probe(self):
        pass

    def isSatisfied(self):
        return self._task.hasBeenPerformed()

    def errorMessage(self):
        return "Task has not been performed"


class RunningNumberGenerator(object):

    def __init__(self):
        self._nextSeqNo= 0
        self._lock= threading.RLock()

    def nextNumber(self):
        self._lock.acquire()
        result= self._nextSeqNo
        self._nextSeqNo+= 1
        self._lock.release()

        return result


class DummyTask(Task):

    def __init__(self, sequenceNumberGenerator=RunningNumberGenerator(),
                 name="dummy"):
        super(DummyTask, self).__init__()
        self._sequenceNumberGenerator= sequenceNumberGenerator
        self._name= name

        self._runningNumber= None
        self._performed= False
        self._mutex= threading.Lock()


    @override
    def name(self):
        return self._name


    @synchronized("_mutex")
    @override
    def perform(self):
        assert not self._performed
        self._runningNumber= self._sequenceNumberGenerator.nextNumber()
        self._performed= True


    @synchronized("_mutex")
    def hasBeenPerformed(self):
        return self._performed


    def runningNumber(self):
        assert self._runningNumber is not None
        return self._runningNumber


class FailingTask(Task):

    def __init__(self, name=None):
        super(FailingTask, self).__init__()
        self._name= name
        self._taskPerformed= False
        self._mutex= threading.Lock()

    @synchronized("_mutex")
    @override
    def perform(self):
        self._taskPerformed= True
        raise Exception("requested failure")


    @synchronized("_mutex")
    def hasBeenPerformed(self):
        return self._taskPerformed


    @override
    def name(self):
        if self._name is None:
            return "FailingTask"
        else:
            return self._name


class SlowTask(Task):

    def __init__(self, sleepDurationSec, name=""):
        Task.__init__(self)
        self._sleepDurationSec= sleepDurationSec
        self._name= name
        self._performed= False
        self._busy= False
        self._canceled= False
        self._mutex= threading.RLock()


    @override
    def perform(self):
        assert not self._performed

        self._setBusyFlag(True)
        MAX_SLEEP_TIME_PER_PASS_SEC= 0.5
        remainingSleepTimeSec= self._sleepDurationSec
        while remainingSleepTimeSec > 0.0 and not self.isCanceled():
            sleepTimeSec= min(MAX_SLEEP_TIME_PER_PASS_SEC,
                              self._sleepDurationSec)
            time.sleep(sleepTimeSec)
            remainingSleepTimeSec-= sleepTimeSec
        self._setBusyFlag(False)
        self._onPerformed()


    @override
    def name(self):
        return self._name


    @synchronized("_mutex")
    def _onPerformed(self):
        self._performed= True


    @synchronized("_mutex")
    def _setBusyFlag(self, busy):
        self._busy= busy


    @synchronized("_mutex")
    def isBusy(self):
        return self._busy


    @synchronized("_mutex")
    def hasBeenPerformed(self):
        return self._performed


    @synchronized("_mutex")
    def isCanceled(self):
        return self._canceled


    @synchronized("_mutex")
    def cancel(self):
        self._canceled= True


class SingleThreadExecutorTest(unittest.TestCase):


    def setUp(self):
        logger= DummyLogger()
        self.executor= SingleThreadExecutor(logger)
        self._seqNoGenerator= RunningNumberGenerator()


    def test_optimistic(self):
        t= self._createDummyTask()
        self.executor.execute(t)
        Poller(5).check(TaskCompletionProbe(t))
        self.assertTrue(t.hasBeenPerformed())


    def test_pessimistic_task(self):
        task_1= FailingTask()
        task_2= self._createDummyTask()
        self.executor.execute(task_1)
        self.executor.execute(task_2)
        Poller(5).check(TaskCompletionProbe(task_1))
        Poller(5).check(TaskCompletionProbe(task_2))


    def test_FIFO_like_behaviour(self):
        task_1= self._createDummyTask()
        task_2= self._createDummyTask()
        task_3= self._createDummyTask()
        self.executor.execute(task_1)
        self.executor.execute(task_2)
        self.executor.execute(task_3)
        Poller(5).check(TaskCompletionProbe(task_1))
        Poller(5).check(TaskCompletionProbe(task_2))
        Poller(5).check(TaskCompletionProbe(task_3))
        self.assertTrue(task_1.runningNumber() < task_2.runningNumber())
        self.assertTrue(task_2.runningNumber() < task_3.runningNumber())


    def _createDummyTask(self):
        return DummyTask(self._seqNoGenerator)


    def test_should_return_task_control(self):
        class LongTask(Task):

            def __init__(self, sleepDurationSec):
                self._sleepDurationSec= sleepDurationSec
                self._done= False

            @override
            def perform(self):
                while not self._done:
                    time.sleep(self._sleepDurationSec)

            def terminate(self):
                self._done= True

        task= LongTask(0.01)
        taskCtrl= self.executor.execute(task)
        self.assertFalse(taskCtrl.isDone())
        Poller(1, pollingDelaySec=0.01).check(ExecutionProbe(
            lambda: self.assertTrue(taskCtrl.isRunning())))
        task.terminate()
        Poller(2, pollingDelaySec=0.01).check(
            ExecutionProbe(lambda: self.assertTrue(taskCtrl.isDone())))

        task= LongTask(0.5)
        taskCtrl= self.executor.execute(task)
        task.terminate()
        taskCtrl.waitForCompletion()
        self.assertTrue(taskCtrl.isDone())


class SingleThreadImmediateExecutorTest(unittest.TestCase):


    def setUp(self):
        logger= DummyLogger()
        self.wrappedExecutor= SingleThreadExecutor(logger)
        self.executor= SingleThreadImmediateExecutor(self.wrappedExecutor)


    def test_construction(self):
        self.assertTrue(self.executor.isIdle())


    def test_should_discard_task_when_executor_is_busy(self):

        class SlowTaskBusyProbe(Probe):
            def __init__(self, task_1):
                Probe.__init__(self)
                self._task= task_1

            @override
            def probe(self):
                pass

            @override
            def isSatisfied(self):
                return self._task.isBusy()

            @override
            def errorMessage(self):
                return "Task is not busy"


        class ExecutorIdleProbe(Probe):
            def __init__(self, executor):
                Probe.__init__(self)
                self.executor= executor

            @override
            def probe(self):
                pass

            @override
            def isSatisfied(self):
                return self.executor.isIdle()

            @override
            def errorMessage(self):
                return "Executor is busy"


        SLEEP_DURATION_SEC= 1.0
        task_1= SlowTask(sleepDurationSec=SLEEP_DURATION_SEC, name="one")

        self.executor.execute(task_1)
        Poller(4).check(SlowTaskBusyProbe(task_1))
        time.sleep(SLEEP_DURATION_SEC / 5.0)

        task_2= DummyTask("two")
        try:
            self.executor.execute(task_2)
        except BusyExecutorException as e:
            self.assertEqual("one", e.nameOfCurrentTask)

        Poller(4).check(ExecutorIdleProbe(self.executor))
        self.executor.execute(task_2)


    def test_should_ignore_failing_tasks(self):
        failingTask= FailingTask("broken")
        self.executor.execute(failingTask)
        Poller(4).check(TaskCompletionProbe(failingTask))


        dummyTask= SlowTask(0, "slow task")
        Poller(1).check(ExecutionProbe(lambda:
                        self.executor.execute(dummyTask)))
        Poller(4).check(TaskCompletionProbe(dummyTask))


    def test_thread_safetyness(self):
        if TestHelper.areLongRunningTestsInhibited():
            logging.warning("Skipping test because"
                            " long running tests are inhibited")
            return

        class UnsynchronziedSerialExecutor(Executor):
            def __init__(self):
                Executor.__init__(self)
                self._nInvocations= 0


            @override
            def execute(self, task):
                newValue= self._nInvocations + 1
                task.perform()
                self._nInvocations= newValue
                return DoneTaskControl()


            def nInvocations(self):
                return self._nInvocations


        class StressThread(threading.Thread):

            def __init__(self, executor, task, nPasses):
                threading.Thread.__init__(self)
                self._executor= executor
                self._task= task
                self._nPasses= nPasses
                self._nSuccessfullInvocations= 0


            def run(self):
                logging.debug("%s: run" % t.name)
                for _ in range(0, self._nPasses):
                    self._nSuccessfullInvocations+= 1
                    try:
                        self._executor.execute(self._task)
                    except:
                        time.sleep(random.random()* 0.01)
                        self._nSuccessfullInvocations-= 1


            def nSuccessFullInvocations(self):
                return self._nSuccessfullInvocations

        self.wrappedExecutor= UnsynchronziedSerialExecutor()
        self.executor= SingleThreadImmediateExecutor(self.wrappedExecutor)

        N_THREADS= 8
        N_PASSES= 1000
        threads= []

        for _ in range(0, N_THREADS):
            task= SlowTask(0)
            t= StressThread(self.executor, task, N_PASSES)
            threads.append(t)

        for t in threads:
            logging.debug("Starting " + t.name)
            t.start()

        totalSuccessFullInvocations= 0
        for t in threads:
            logging.debug("Waiting until all threads are done")
            t.join()
            logging.debug("#successFullinvocations(%s): %d" %(
                t.name, t.nSuccessFullInvocations()))
            totalSuccessFullInvocations+= t.nSuccessFullInvocations()

        self.assertEqual(totalSuccessFullInvocations,
                         self.wrappedExecutor.nInvocations())


class FunctionTaskTest(unittest.TestCase):


    def _nulladicFunc(self):
        self._funcInvoked= True


    def test_should_execute_nulladic_function(self):
        self._funcInvoked= False
        t= FunctionTask(self._nulladicFunc)
        t.perform()
        self.assertTrue(self._funcInvoked)


    def test_should_have_name(self):
        t= FunctionTask(self._nulladicFunc)
        self.assertTrue("_nulladicFunc" in t.name())


    def test_should_have_custom_name(self):
        t= FunctionTask(self._nulladicFunc).withName("spam")
        self.assertEqual("spam", t.name())


    def test_should_execute_bounded_method(self):
        if TestHelper.areLongRunningTestsInhibited():
            return

        class Tux(object):

            def fly(self, *args, **kwds):
                self._args= args
                self._kwds= kwds
                self._funcInvoked= True


        executor= MultiThreadExecutor(
            maxThreads=10, logger=DummyLogger())
        tuxes= []
        futures= []
        for _ in range(0, 1000):
            tux= Tux()
            tuxes.append(tux)
            t= FunctionTask(tux.fly, 1, 2, "foobar", foo="bar")
            future= executor.execute(t)
            futures.append(future)

        for each in futures:
            each.waitForCompletion()

        for eachTux in tuxes:
            self.assertEqual((1, 2, "foobar"), eachTux._args)
            self.assertEqual({"foo": "bar"}, eachTux._kwds)
            self.assertTrue(eachTux._funcInvoked)


    def _method(self, l, d, lock):
        with lock:
            d["foo"]= 1
            l.append("foo")


    def test_should_execute_self_method(self):
        if TestHelper.areLongRunningTestsInhibited():
            return

        executor= MultiThreadExecutor(
            maxThreads=10, logger=DummyLogger())
        N_TASKS= 5000
        results= []
        d= {}
        lock= threading.Lock()
        futures= []
        for _ in range(0, N_TASKS):
            t= FunctionTask(self._method, results, d, lock)
            future= executor.execute(t)
            futures.append(future)

        for each in futures:
            each.waitForCompletion()

        self.assertEqual(N_TASKS, len(results))
        for each in results:
            self.assertEqual("foo", each)


    def test_should_provide_exception(self):
        class CustomException(Exception):
            pass

        def fail():
            raise CustomException("don't panic")
        task= FunctionTask(fail)
        self.assertRaises(CustomException, task.perform)
        self.assertEqual("don't panic", str(task.getException()))
        self.assertRaises(CustomException, task.reraiseExceptionIfAny)

        def okay():
            pass
        task= FunctionTask(okay)
        task.perform()
        self.assertTrue(task.getException() is None)
        task.reraiseExceptionIfAny()


    def test_should_provide_result(self):
        def f():
            return 42
        task= FunctionTask(f)
        self.assertRaises(FunctionTaskException, task.getResult)
        task.perform()
        self.assertEqual(42, task.getResult())


class MultiThreadExecutorTest(unittest.TestCase):


    def setUp(self):
        logger= DummyLogger()
        self.executor= MultiThreadExecutor(3, logger)


    def test_should_use_multiple_threads(self):
        if TestHelper.areLongRunningTestsInhibited():
            return

        sleepDurationSec= 2
        f1= self.executor.execute(SlowTask(sleepDurationSec))
        f2= self.executor.execute(SlowTask(sleepDurationSec))
        f3= self.executor.execute(SlowTask(sleepDurationSec))

        upperExecutionLimit= 1.5 * sleepDurationSec
        t0= time.time()
        for each in [f1, f2, f3]:
            each.waitForCompletion()
        t1= time.time()
        self.assertTrue(t1 - t0 < upperExecutionLimit)



    def test_should_raise_timeout_error(self):
        task= SlowTask(sleepDurationSec=100.0)
        taskCtrl= self.executor.execute(task)
        self.assertRaises(TaskTimeoutError,
                          taskCtrl.waitForCompletion, timeoutSec=0.01)
        task.cancel()


class LogTaskFailureDecoratorTest(unittest.TestCase):

    def test_logging(self):

        class MyLoggerListener(LoggerListener):

            def __init__(self):
                self.errorCount= 0


            def onError(self, message):
                self.errorCount+= 1


        class TestTask(Task):
            def __init__(self, logger):
                Task.__init__(self)
                self._logger= logger

            @logTaskFailure
            def perform(self):
                raise Exception("expected failure")


        logger= ObservableLogger(DummyLogger())
        listener= MyLoggerListener()
        self.assertRaises(Exception,
                          TestTask(logger).perform)
        self.assertTrue(1, listener.errorCount)


class TaskDelayingExecutorTest(unittest.TestCase):

    def test_delay(self):
        class FooTask(Task):
            def __init__(self, results):
                super(FooTask, self).__init__()
                self._results= results

            @override
            def perform(self):
                self._results.append(23)

        taskResult= []
        task= FooTask(taskResult)
        exe= TaskDelayingExecutor()
        taskControl= exe.execute(task)

        tasks= exe.getDelayedTasks()
        self.assertEqual(1, len(tasks))
        self.assertEqual(0, len(taskResult))
        self.assertFalse(taskControl.isDone())
        self.assertFalse(taskControl.isRunning())

        exe.executeDelayedTasks()
        self.assertEqual(23, taskResult[0])
        self.assertTrue(taskControl.isDone())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
