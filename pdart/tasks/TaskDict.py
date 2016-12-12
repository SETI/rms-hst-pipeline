"""
A data structure to hold running :class:`~pdart.tasks.Task.Task` s and
the :class:`~pdart.tasks.TaskProcess.TaskProcess` es in which they're
running.  :class:`~pdart.tasks.TaskDict.TaskDict` is a part of
:class:`~pdart.tasks.TaskQueue.TaskQueue`.
"""
import pdart.tasks.TaskProcess

from pdart.tasks.Task import Task  # for mypy
from typing import Dict, Sequence, Set  # for mypy


class TaskDict(dict):
    """
    A Python dictionary with :class:`~pdart.tasks.Task.Task` keys and
    :class:`~pdart.tasks.TaskProcess.TaskProcess` values plus
    functionality to automatically create the process values when
    inserting task keys.
    """
    def __init__(self):
        # type: () -> None
        pass

    def insert_task(self, task):
        # type: (Task) -> None
        """
        Insert a new task to run into the
        :class:`~pdart.tasks.TaskDict.TaskDict` automatically
        launching the task and creating the corresponding
        :class:`~pdart.tasks.TaskProcess.TaskProcess` values for the
        key.  The task must not already be in the
        :class:`~pdart.tasks.TaskDict.TaskDict`.
        """
        assert task
        assert task not in self, ('Duplicate task: %s' % task)
        self[task] = pdart.tasks.TaskProcess.TaskProcess(task)

    def insert_tasks(self, tasks):
        # type: (Sequence[Task]) -> None
        """
        Insert a list of new tasks to run into the
        :class:`~pdart.tasks.TaskDict.TaskDict`, automatically
        launching the tasks and creating the corresponding
        :class:`~pdart.tasks.TaskProcess.TaskProcess` values for the
        keys.  The tasks must not already be in the
        :class:`~pdart.tasks.TaskDict.TaskDict`.
        """
        assert tasks
        for task in tasks:
            self.insert_task(task)

    def isdisjoint(self, other_tasks):
        # type: (Set[Task]) -> bool
        """
        Return ``True`` if the set of keys for this dictionary is
        disjoint from the argument.
        """
        for task in other_tasks:
            if task in self:
                return False
        return True
