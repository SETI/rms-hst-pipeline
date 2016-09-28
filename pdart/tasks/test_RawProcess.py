from pdart.tasks.RawProcess import *
from pdart.tasks.NullTask import NullTask
from pdart.tasks.TaskProcess import TaskProcess

import os
import signal
import time


# TODO Some of these tasks hang when they fail.  Set up an automatic
# clean-up for them so we don't hang the whole test suite.


def _a_minute_from_now():
    return time.time() + 60


def test_exit_code_to_status():
    assert exit_code_to_status(None) == 'RUNNING'
    assert exit_code_to_status(0) == 'SUCCEEDED'
    assert exit_code_to_status(-1) == 'TERMINATED'
    assert exit_code_to_status(1) == 'FAILED'


def _pass():
    pass


def run_test_Process_returning(proc):
    # Test one: process immediately returns
    assert proc
    assert not proc.was_started()
    assert not proc.is_alive()
    assert proc.status() == 'INITIALIZED'

    proc.start()
    assert proc.was_started()
    # might still be running; might not
    stat = proc.status()
    assert stat in ['RUNNING', 'SUCCEEDED']
    if stat == 'SUCCEEDED':
        assert not proc.is_alive()

    proc.join()
    # is definitely not running
    stat = proc.status()
    assert stat == 'SUCCEEDED'
    assert not proc.is_alive()


def test_RawProcess_returning():
    proc = RawProcess(target=_pass, args=tuple())
    run_test_Process_returning(proc)


def test_TerminatableProcess_returning():
    proc = TerminatableProcess(target=_pass, args=tuple())
    run_test_Process_returning(proc)


def test_TimeoutProcess_returning():
    proc = TimeoutProcess(_a_minute_from_now(), _pass, tuple())
    run_test_Process_returning(proc)


class PassTask(NullTask):
    def __init__(self):
        NullTask.__init__(self)

    def __str__(self):
        return 'PassTask'

    def run(self):
        _pass()


def test_TaskProcess_returning():
    proc = TaskProcess(PassTask())
    run_test_Process_returning(proc)


def _sleep(secs):
    time.sleep(secs)


def run_test_Process_terminate(proc):
    # Test two: process hangs
    proc.start()
    assert proc.status() == 'RUNNING'
    proc.terminate()
    proc.join()
    assert proc.status() == 'TERMINATED'


def test_RawProcess_terminate():
    proc = RawProcess(target=_sleep, args=(60,))
    run_test_Process_terminate(proc)


def test_TerminatableProcess_terminate():
    proc = TerminatableProcess(target=_sleep, args=(60,))
    run_test_Process_terminate(proc)


def test_TimeoutProcess_terminate():
    proc = TimeoutProcess(_a_minute_from_now(), _sleep, (60,))
    run_test_Process_terminate(proc)


class SleepTask(NullTask):
    def __init__(self):
        NullTask.__init__(self)

    def __str__(self):
        return 'SleepTask'

    def run(self):
        _sleep()


def test_TaskProcess_terminate():
    proc = TaskProcess(SleepTask())
    run_test_Process_terminate(proc)


def _nap():
    while True:
        time.sleep(10)


_JOIN_SECS = 10 * 1.0 / 1000  # 10 ms


def run_test_Process_terminating(proc):
    # Test three: process takes a while to terminate.  Check that it
    # shows 'TERMINATING' status before it goes to 'TERMINATED'.

    # Try to end the process (but it'll ignore the request and keep on
    # going)
    proc.terminate()
    assert proc.status() == 'TERMINATING'
    proc.join(_JOIN_SECS)
    # it still hasn't terminated even after we wait
    assert proc.status() == 'TERMINATING'

    # Send a different signal to terminate it with extreme prejudice
    os.kill(proc.process.pid, signal.SIGKILL)

    proc.join(_JOIN_SECS)
    # Now it's dead
    assert proc.status() == 'TERMINATED'


def test_TerminatableProcess_terminating():
    # The signal handler needs to be set *before* we fork the
    # process
    original_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    proc = TerminatableProcess(target=_nap, args=tuple())
    proc.start()
    signal.signal(signal.SIGTERM, original_handler)

    run_test_Process_terminating(proc)


def test_TimeoutProcess_terminating():
    # The signal handler needs to be set *before* we fork the
    # process
    original_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    proc = TimeoutProcess(_a_minute_from_now(), _nap, tuple())
    proc.start()
    signal.signal(signal.SIGTERM, original_handler)

    run_test_Process_terminating(proc)


class NapTask(NullTask):
    def __init__(self):
        NullTask.__init__(self)

    def __str__(self):
        return 'NapTask'

    def run(self):
        _nap()


def test_TaskProcess_terminating():
    # The signal handler needs to be set *before* we fork the
    # process
    original_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    proc = TaskProcess(NapTask())
    proc.start()
    signal.signal(signal.SIGTERM, original_handler)

    run_test_Process_terminating(proc)


def run_test_timing_out(proc):
    # proc times out in half a second

    # a full second later
    time.sleep(1)
    assert proc.status() == 'TIMING_OUT'
    proc.join(_JOIN_SECS)
    # it still hasn't terminated even after we wait
    assert proc.status() == 'TIMING_OUT'

    # Send a different signal to terminate it with extreme prejudice
    os.kill(proc.process.pid, signal.SIGKILL)

    proc.join(_JOIN_SECS)
    # Now it's dead
    assert proc.status() == 'TIMED_OUT'


def test_TimeoutProcess_timing_out():
    # times out in half a second
    original_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    proc = TimeoutProcess(time.time() + 0.5, _nap, tuple())
    proc.start()
    signal.signal(signal.SIGTERM, original_handler)

    run_test_timing_out(proc)


def test_TaskProcess_timing_out():
    # times out in half a second
    original_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    task = NapTask()
    task.deadline_time = time.time() + 0.5
    proc = TaskProcess(task)
    proc.start()
    signal.signal(signal.SIGTERM, original_handler)

    run_test_timing_out(proc)
