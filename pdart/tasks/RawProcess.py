"""
Contains a chain of classes representing external system processes in
which :class:`~pdart.tasks.Task.Task` s are run.  We start with
Python's :class:`multiprocessing.Process`, wrap it, then incrementally
add functionality.

We could have written code directly using
:class:`multiprocessing.Process`, but be incrementally wrapping and
adding functionality, we can test each layer of functionality
separately.
"""
import multiprocessing
import time

from typing import Callable, Tuple


def exit_code_to_status(exit_code):
    # type: (int) -> str
    """
    Convert the exit code in :attr:`multiprocessing.Process.exitcode`
    into a status string.
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
    A wrapper around a Python :class:`multiprocessing.Process` with a
    bit more instrumentation.
    """
    def __init__(self, target, args):
        # type: (Callable, Tuple) -> None
        assert target
        assert args is not None
        self.process = multiprocessing.Process(target=target, args=args)

    def start(self):
        # type: () -> None
        """
        Launch the :class:`multiprocessing.Process` in another system
        process.
        """
        assert not self.was_started()
        self.process.start()

    def was_started(self):
        # type: () -> bool
        """
        Return ``True`` iff
        :meth:`~pdart.tasks.RawProcess.RawProcess.start` was called.
        """
        return self.process.pid is not None

    def is_alive(self):
        # type: () -> bool
        """
        Return ``True`` iff the process was started and has not yet
        completed.
        """
        return self.process.is_alive()

    def status(self):
        # type: () -> str
        """
        Return one of ``INITIALIZED``, ``RUNNING``, ``SUCCEEDED``,
        ``FAILED``, or ``TERMINATED``.
        """
        if self.process.pid is None:
            return 'INITIALIZED'
        return exit_code_to_status(self.process.exitcode)

    def join(self, timeout=None):
        # type: (float) -> None
        """
        Wait for the given amount of time for the process to complete.
        If the argument is ``None``, wait indefinitely.
        """
        assert self.was_started()
        self.process.join(timeout)

    def terminate(self):
        # type: () -> None
        """
        Try to terminate the process by sending a ``SIGTERM`` signal.
        """
        self.process.terminate()


class TerminatableProcess(RawProcess):
    """
    A :class:`~pdart.tasks.RawProcess.RawProcess` plus tracking
    whether the process been terminated or not.
    """
    def __init__(self, target, args):
        # type: (Callable, Tuple) -> None
        RawProcess.__init__(self, target, args)
        self.terminating = False

    def terminate(self):
        if not self.terminating:
            self.terminating = True
            RawProcess.terminate(self)

    def status(self):
        """
        Like :class:`~pdart.tasks.RawProcess.RawProcess`, but will
        return ``TERMINATING`` if ``terminate()`` has been called but
        the process has not yet completed.
        """
        status = RawProcess.status(self)
        if self.terminating:
            if status == 'RUNNING':
                status = 'TERMINATING'
        return status


class TimeoutProcess(TerminatableProcess):
    """
    A :class:`~pdart.tasks.RawProcess.TerminatableProcess` plus a time
    limit and tracking on whether it's timed out or not.
    """
    def __init__(self, deadline_time, target, args):
        # type: (float, Callable, Tuple) -> None
        TerminatableProcess.__init__(self, target, args)
        assert deadline_time
        self.timing_out = False
        self.deadline_time = deadline_time

    def status(self):
        """
        Like :class:`~pdart.tasks.RawProcess.TerminatableProcess`, but
        checks whether the process has exceeded its timeout.  If so it
        will return ``TIMING_OUT`` and ``TIMED_OUT`` instead of
        ``TERMINATING`` and ``TERMINATED``.
        """
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
        # type: () -> None
        """
        Mark the process as timed out and terminate it.
        """
        self.timing_out = True
        self.terminate()
