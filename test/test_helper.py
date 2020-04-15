
import abc
import io
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import unittest


ENV_VAR_LONG_RUNNING_TESTS_ENABLE = "ENABLE_LONG_RUNNING_TESTS"
XVFB_DEFAULT_DISPLAY = ":42"


def setUpLogger(loggingLevel):
    FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
    f = logging.Formatter(fmt=FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(loggingLevel)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(f)
        handler.setLevel(loggingLevel)
        root_logger.addHandler(handler)


class TestHelper(object):

    TIMEOUT_FACTOR = 1.0

    @staticmethod
    def sleep(durationSec):
        time.sleep(durationSec)

    @staticmethod
    def longRunningTest(f):
        def wrappedMethod(self, *args, **kwds):
            if TestHelper.areLongRunningTestsInhibited():
                TestHelper._logSkippedTest()
                return
            else:
                return f(self, *args, **kwds)

        return wrappedMethod

    @staticmethod
    def areLongRunningTestsInhibited():
        return ENV_VAR_LONG_RUNNING_TESTS_ENABLE not in os.environ

    @staticmethod
    def _logSkippedTest():
        logging.warning("Test is skipped because long running tests"
                        " are inhibited.  Set environment variable %s"
                        " in order to enable this test!" % (
                            ENV_VAR_LONG_RUNNING_TESTS_ENABLE))

    @staticmethod
    def areDictEqual(d1, d2):
        items1 = list(d1.items())
        items2 = list(d2.items())
        if len(items1) != len(items2):
            return False

        for each in d1:
            if d1[each] != d2[each]:
                return False

        return True

    @staticmethod
    def whenLongRunningTestsAreInhibitedThenExitWithSuccess():
        if TestHelper.areLongRunningTestsInhibited():
            TestHelper._logSkippedTest()
            sys.exit(os.EX_OK)

    @staticmethod
    def checkThatFileContainsString(logFilePath, expectedMessage,
                                    timeoutInSec=5.0):
        WAIT_END = time.time() + timeoutInSec
        while not TestHelper.isFileContainingString(expectedMessage,
                                                    logFilePath):
            time.sleep(0.05)
            if time.time() > WAIT_END:
                raise RuntimeError(("File %s is lacking string '%s' "
                                    "after waiting for %.3f seconds" % (
                                        logFilePath, expectedMessage,
                                        timeoutInSec)))

    @staticmethod
    def isFileContainingString(expectedMessage, filePath):
        try:
            return 1 <= TestHelper.getNumberOfMatches(
                re.escape(expectedMessage)).inFile(filePath)
        except:
            return False

    @staticmethod
    def writeLinesToFile(filePath, fileContentsAsLines):
        f = open(filePath, "wb")
        for each in fileContentsAsLines:
            f.write(each)
            f.write('\n')
        f.close()

    @staticmethod
    def loggingTurnedOffMessageForPort(port):
        return "Logging is turned off for setting output port %d" % port

    @staticmethod
    def startService(serviceCommand, serviceLogPath, startUpMessage):
        assert 0 < len(serviceCommand)
        assert 0 < len(serviceLogPath)

        return TestHelper.startServiceWithArgs(
            shlex.split(serviceCommand),
            serviceLogPath,
            startUpMessage)

    @staticmethod
    def startServiceWithArgs(arguments, serviceLogPath, startUpMessage,
                             timeoutSec=7):
        assert len(arguments) > 0
        assert len(serviceLogPath) > 0

        serviceLog = open(serviceLogPath, "wb")
        try:
            proc = subprocess.Popen(arguments,
                                    stdout=serviceLog,
                                    stderr=serviceLog)
        except Exception as e:
            raise Exception("Failed to execute '%s' (%s)" % (
                            arguments, str(e)))

        Poller(timeoutSec, pollingDelaySec=0.2).check(
            MessageInFileProbe(startUpMessage, serviceLogPath))

        return proc

    @staticmethod
    def isSubprocessAlive(process):
        return process.poll() is None

    @staticmethod
    def terminateSubprocess(process):
        exitCode = process.poll()
        if exitCode is not None:
            logging.info("Process with PID %d already dead" % process.pid)
            return
        TestHelper._terminateByUsingPid(process)
        END_TIME = time.time() + 5.0
        done = False
        while not done:
            if process.poll() is not None:
                done = True
                break
            elif time.time() > END_TIME:
                assert False, "Process with PID %d failed to die" % process.pid

        assert process.returncode is not None

    @staticmethod
    def _terminateByUsingPid(process):
        starterPid = process.pid
        if os.EX_OK == subprocess.call("kill -INT %d" % starterPid,
                                       shell=True):
            try:
                Poller(5).check(SubprocessTerminated(process, "<no name>"))
                return
            except:
                pass

    @staticmethod
    def dumpFileToStdout(filePath):
        assert 0 < len(filePath)
        if not os.path.exists(filePath):
            print("File %s is missing" % filePath)
        else:
            print("--- BEGIN %s ---" % filePath)
            with io.open(filePath, "r") as f:
                for eachLine in f:
                    print(eachLine, end=' ')
            print("--- END %s ---" % filePath)
        sys.stdout.flush()

    @staticmethod
    def executeShellCommand(cmd):
        exitCode = subprocess.call(cmd, shell=True)
        if os.EX_OK != exitCode:
            raise Exception("Failed to execute shell command '%s'."
                            "  Exit code is: %d" % (cmd, exitCode))

    @staticmethod
    def removeFileIfAny(path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def getNumberOfMatches(pattern):
        class MatchCounter(object):
            def __init__(self, pattern):
                self._pattern = pattern

            def inFile(self, filePath):
                with io.open(filePath, 'r') as fd:
                    content = fd.read()
                res = re.findall(self._pattern, content)
                return len(res)

        return MatchCounter(pattern)

    @staticmethod
    def getNumberOfMessages(message):
        return TestHelper.getNumberOfMatches(re.escape(message))

    @staticmethod
    def executeQtGUITestsWithXvfb(testClass, argv, XvfbLogPath='./xvfb.log'):
        result = None
        qtApp = None
        from PyQt5 import QtWidgets
        with Xvfb.launch(XvfbLogPath) as xvfb:
            qtApp = QtWidgets.QApplication(argv)
            result = unittest.TextTestRunner().run(
                unittest.TestLoader().loadTestsFromTestCase(testClass))
            assert qtApp is not None
            del qtApp

        assert result is not None
        if not result.wasSuccessful():
            sys.exit(os.EX_SOFTWARE)

    @staticmethod
    def qtPoll(timeoutSec):
        return _QtPollCheck(timeoutSec)


class _QtPollCheck(object):

    def __init__(self, timeoutSec):
        self._timeoutSec = timeoutSec

    def check(self, f, *args, **kwds):
        def _check():
            from PyQt5.Qt import QApplication
            QApplication.processEvents()
            f(*args, **kwds)

        Poller(self._timeoutSec).check(ExecutionProbe(_check))
        return self


def _pathRelativeToThisFile(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)


class Probe(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def probe(self):
        assert False, "Abstract method called"

    @abc.abstractmethod
    def isSatisfied(self):
        assert False, "Abstract method called"

    @abc.abstractmethod
    def errorMessage(self):
        assert False, "Abstract method called"


class Poller(object):

    def _adjustTimeoutIfComputerIsSlow(self):
        if TestHelper.TIMEOUT_FACTOR != 1.0:
            adjustedTimeoutSec = (
                self._timeoutSec * TestHelper.TIMEOUT_FACTOR)
            logging.info(
                "Increasing polling timeout from %.1f to %.1f sec"
                " because computer performance factor is set to %.2f." % (
                    self._timeoutSec, adjustedTimeoutSec,
                    TestHelper.TIMEOUT_FACTOR))
            self._timeoutSec = adjustedTimeoutSec

    def __init__(self, timeoutSec, pollingDelaySec=0.01, reportPeriodSec=1.0):
        assert timeoutSec >= 0
        self._timeoutSec = timeoutSec
        self._pollingDelaySec = pollingDelaySec
        self._reportPeriodSec = reportPeriodSec
        self._nextReportTimeSec = None
        self._t0 = None

        self._adjustTimeoutIfComputerIsSlow()

    def _reportProgressIfAny(self):
        now = time.time()
        if self._nextReportTimeSec is None:
            self._nextReportTimeSec = now + self._reportPeriodSec

        if time.time() > self._nextReportTimeSec:
            logging.info("Polling for %.1f sec.  Timeout is %.1f sec" % (
                now - self._t0, self._timeoutSec))
            self._nextReportTimeSec += self._reportPeriodSec

    def check(self, probe):
        self._t0 = time.time()
        END_TIME_SEC = self._t0 + self._timeoutSec
        done = False
        while not done:
            self._reportProgressIfAny()
            probe.probe()
            if probe.isSatisfied():
                done = True
            elif time.time() >= END_TIME_SEC:
                raise RuntimeError(
                    "Timeout of %s seconds while polling (%s)" % (
                        self._timeoutSec, probe.errorMessage()))
            else:
                time.sleep(self._pollingDelaySec)


class CommandOutputProbe(Probe):

    def __init__(self, shellCommand, expectedStringInOutput):
        Probe.__init__(self)
        self._shellCmd = shellCommand
        self._expectedString = expectedStringInOutput
        self._satisfied = False

    def probe(self):
        process = subprocess.Popen(shlex.split(self._shellCmd),
                                   stdout=subprocess.PIPE)
        out, _ = process.communicate()
        if self._expectedString in out:
            self._satisfied = True

    def isSatisfied(self):
        return self._satisfied

    def errorMessage(self):
        return "Failed to find '%s' in output of command '%s'" % (
               self._expectedString, self._shellCmd)


class SuccessfullShellCommandProbe(Probe):

    def __init__(self, command):
        Probe.__init__(self)
        self._command = command

    def probe(self):
        self._exitCode = os.system(self._command)

    def isSatisfied(self):
        return os.EX_OK == self._exitCode

    def errorMessage(self):
        return "Failed to run '%s'" % (self._command)


class MessageInFileProbe(Probe):

    def __init__(self, expectedMessage, filePath):
        self._expectedMessage = expectedMessage
        self._filePath = filePath

    def probe(self):
        pass

    def isSatisfied(self):
        return TestHelper.isFileContainingString(self._expectedMessage,
                                                 self._filePath)

    def errorMessage(self):
        return "Message '%s' is missing in file %s" % (self._expectedMessage,
                                                       self._filePath)


class SubprocessTerminated(Probe):

    def __init__(self, process, name):
        self._process = process
        self._name = name

    def probe(self):
        self._process.poll()

    def isSatisfied(self):
        return self._process.returncode is not None

    def errorMessage(self):
        return "Subprocess '%s' is not terminated" % self._name


class ExecutionProbe(Probe):

    def __init__(self, func, name="", *args, **kwds):
        self._func = func
        self._succeeded = False
        self._name = name
        self._args = args
        self._kwds = kwds
        self._lastException = None
        self._lastTraceback = None

    @staticmethod
    def create(func, *args, **kwds):
        return ExecutionProbe(func, "no_name", *args, **kwds)

    def probe(self):
        try:
            self._func(*self._args, **self._kwds)
            self._succeeded = True
        except Exception as e:
            self._lastException = e
            self._lastTracebackFormat = traceback.format_exc()
            logging.debug(e)

    def isSatisfied(self):
        return self._succeeded

    def errorMessage(self):
        return ("Failed to execute function %s (%s). "
                "Last exception: '%s' (%s)") % (
                    self._func.__name__, self._name, str(self._lastException),
                    self._lastTracebackFormat)


class PollingTestTask(object):

    def __init__(self, probe, timeoutSec):
        self._probe = probe
        self._timeoutSec = timeoutSec
        self._startTime = None
        self._waitEndTime = None

    def hasPassed(self):
        return self._probe.isSatisfied()

    def check(self):
        if self._startTime is None:
            self._startTime = time.time()
            self._waitEndTime = self._startTime + self._timeoutSec

        self._probe.probe()
        if self._probe.isSatisfied():
            pass
        elif time.time() >= self._waitEndTime:
            raise RuntimeError(
                "Timeout of %s seconds occured while polling (%s)" % (
                    self._timeoutSec, self._probe.errorMessage()))


class TestSuiteTask(object):

    def __init__(self, testTasks):
        self._testTasks = testTasks

    def check(self):
        try:
            self._check()
        except Exception as e:
            logging.error(str(e))
            sys.exit(1)

    def _check(self):
        nPassedTasks = 0
        for each in self._testTasks:
            each.check()
            if each.hasPassed():
                nPassedTasks += 1

        totalTests = len(self._testTasks)
        if totalTests == nPassedTasks:
            logging.info("%d of %d tests have passed" % (
                nPassedTasks, totalTests))
            sys.exit(os.EX_OK)


def terminateSubprocess(process):
    TestHelper.terminateSubprocess(process)


def isSubprocessAlive(process):
    return TestHelper.isSubprocessAlive(process)


def dumpFileToStdout(filePath):
    TestHelper.dumpFileToStdout(filePath)


class Xvfb(object):

    XVFB_PATH = "/usr/bin/Xvfb"
    X_SERVER_CHECK_COMMAND_LOG_PATH = "./xdpyinfo.log"
    X_SERVER_CHECK_COMMAND = "xdpyinfo >%s 2>&1" % (
        X_SERVER_CHECK_COMMAND_LOG_PATH)

    @staticmethod
    def launch(xvfbLogFilePath):
        os.environ["DISPLAY"] = XVFB_DEFAULT_DISPLAY
        xvfb = Xvfb(xvfbLogFilePath, XVFB_DEFAULT_DISPLAY)
        xvfb.start()

        return xvfb

    def __init__(self, xvfbLogFilePath, displayId):
        self._xvfbLogFilePath = xvfbLogFilePath
        self._displayId = displayId

        if not self._isExe(Xvfb.XVFB_PATH):
            logging.error("Xvfb does not exist in '%s'!" % Xvfb)
            sys.exit(1)

    def _isExe(self, fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    def start(self):
        self._xvfbLogFile = open(self._xvfbLogFilePath, "wb")
        xvfbStartCommand = "%s %s -screen 0 1024x768x24" % (
            Xvfb.XVFB_PATH, self._displayId)
        self._xvfb = subprocess.Popen(
            shlex.split(xvfbStartCommand),
            stdout=self._xvfbLogFile, stderr=self._xvfbLogFile)
        self._waitUntilXvfbIsReady()

    def _waitUntilXvfbIsReady(self):

        class XvfbStartupProbe(Probe):

            def __init__(self, displayId):
                self._displayId = displayId
                self._done = False

            def probe(self):
                assert self._displayId == os.environ["DISPLAY"]
                exitCode = os.system(Xvfb.X_SERVER_CHECK_COMMAND)
                if os.EX_OK == exitCode:
                    self._done = True

            def isSatisfied(self):
                return self._done

            def errorMessage(self):
                return "Failed to detect Xvfb startup"

        try:
            Poller(timeoutSec=5.0).check(XvfbStartupProbe(self._displayId))
        except:
            TestHelper.dumpFileToStdout(self.X_SERVER_CHECK_COMMAND_LOG_PATH)
            raise

    def stop(self):
        terminateSubprocess(self._xvfb)
        self._xvfbLogFile.close()
        os.remove(self._xvfbLogFilePath)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
