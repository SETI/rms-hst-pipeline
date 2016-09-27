import multiprocessing
import time


def exit_code_to_status(exit_code):
    """
    Convert the multiprocessing.Process.exitcode into a status string.
    """
    if exit_code is None:
        return 'RUNNING'
    elif exit_code == 0:
        return 'SUCCEEDED'
    elif exit_code > 0:
        return 'FAILED'
    elif exit_code < 0:
        return 'TERMINATED'


class RawProcess(object):
    """
    A wrapper around a Python multiprocessing.Process with a bit more
    instrumentation.
    """
    def __init__(self, target, args):
        assert target
        assert args is not None
        self.process = multiprocessing.Process(target=target, args=args)

    def was_started(self):
        return self.process.pid is not None

    def start(self):
        assert not self.was_started()
        self.process.start()

    def is_alive(self):
        return self.process.is_alive()

    def status(self):
        if self.process.pid is None:
            return 'INITIALIZED'
        return exit_code_to_status(self.process.exitcode)

    def join(self, timeout=None):
        assert self.was_started()
        self.process.join(timeout)

    def terminate(self):
        self.process.terminate()


class TerminatableProcess(RawProcess):
    """
    A RawProcess plus tracking whether it's been terminated or not.
    """
    def __init__(self, target, args):
        RawProcess.__init__(self, target, args)
        self.terminating = False

    def terminate(self):
        if not self.terminating:
            self.terminating = True
            RawProcess.terminate(self)

    def status(self):
        status = RawProcess.status(self)
        if self.terminating:
            if status == 'RUNNING':
                status = 'TERMINATING'
        return status


class TimeoutProcess(TerminatableProcess):
    """
    A TerminatableProcess plus a time limit and tracking on whether
    it's timed out or not.
    """
    def __init__(self, deadline_time, target, args):
        TerminatableProcess.__init__(self, target, args)
        assert deadline_time
        self.timing_out = False
        self.deadline_time = deadline_time

    def status(self):
        status = TerminatableProcess.status(self)
        if self.timing_out:
            if status == 'TERMINATING':
                status = 'TIMING_OUT'
            elif status == 'TERMINATED':
                status = 'TIMED_OUT'
        else:
            if status == 'RUNNING' and time.time() > self.deadline_time:
                self.timeout()
                status = 'TIMING_OUT'
        return status

    def timeout(self):
        self.timing_out = True
        self.terminate()
