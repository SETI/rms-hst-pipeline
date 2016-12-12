"""
A wrapper around Python's :class:`multiprocessing.Process` to run
:class:`~pdart.tasks.Task.Task` with functionality to set a timeout
for the process.
"""
from pdart.tasks.RawProcess import *

from pdart.tasks.Task import Task  # for mypy


def _run_task_action(task):
    # type: (Task) -> None
    """
    Call the :meth:`~pdart.tasks.Task.Task.run` method of the given
    task.
    """
    task.run()


class TaskProcess(TimeoutProcess):
    """
    A wrapper around :class:`~pdart.tasks.RawProcess.RawProcess` that
    runs a :class:`~pdart.tasks.Task.Task` 's
    :meth:`~pdart.tasks.Task.Task.run` method instead of an arbitrary
    function.
    """
    def __init__(self, task):
        # type: (Task) -> None
        assert task
        TimeoutProcess.__init__(self, task.deadline_time,
                                target=_run_task_action, args=(task,))
