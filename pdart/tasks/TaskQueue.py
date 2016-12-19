"""
A data structure which stores the pending tasks to be run and the
tasks currently running.  :class:`~pdart.tasks.TaskQueue.TaskQueue` is
a part of :class:`~pdart.tasks.TaskRunner.TaskRunner`.
"""
import collections
from pdart.tasks.TaskDict import *

from typing import Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.tasks.Task import Task


class TaskQueue(object):
    """
    An object that contains :class:`~pdart.tasks.Task.Task` s to be
    run.  It contains a list of
    :attr:`~pdart.tasks.TaskQueue.TaskQueue.pending_tasks` and a
    dictionary of
    :attr:`~pdart.tasks.TaskQueue.TaskQueue.running_tasks` containing
    the tasks and the :class:`~pdart.tasks.TaskProcess.TaskProcess` es
    in which they're running.
    """
    def __init__(self):
        # type: () -> None
        """Create an empty task queue."""
        self.pending_tasks = collections.deque()
        # type: collections.deque[Task]
        self.running_tasks = TaskDict()
        assert not self

    def append_pending(self, task):
        # type: (Task) -> None
        """
        Add a new pending :class:`~pdart.tasks.Task.Task` to the
        queue.  The task must not already be in the queue.
        """
        assert task not in self, ('Duplicate task: %s' % task)
        self.pending_tasks.append(task)
        assert self._tasks_are_disjoint()

    def extend_pending(self, tasks):
        # type: (Sequence[Task]) -> None
        """
        Add a sequence of new pending :class:`~pdart.tasks.Task.Task`
        s to the queue.  None of the tasks may already be in the
        queue, nor may there be duplicates.
        """
        for task in tasks:
            assert task not in self, ('Duplicate task: %s' % task)
        self.pending_tasks.extend(tasks)
        assert self._tasks_are_disjoint()

    def is_pending(self, task):
        # type: (Task) -> bool
        """
        Return ``True`` iff the :class:`~pdart.tasks.Task.Task` is
        pending in the queue.
        """
        return task in self.pending_tasks

    def has_pending_tasks(self):
        # type: () -> bool
        """
        Return ``True`` iff there are :class:`~pdart.tasks.Task.Task`
        s pending in the queue.
        """
        return bool(self.pending_tasks)

    def append_running(self, task):
        # type: (Task) -> None
        """
        Add a new :class:`~pdart.tasks.Task.Task` to run now into the
        queue.  The task must not already be in the queue.
        """
        assert task not in self, ('Duplicate task: %s' % task)
        self.running_tasks.insert_task(task)
        assert self._tasks_are_disjoint()

    def extend_running(self, tasks):
        # type: (Sequence[Task]) -> None
        """
        Add a sequence of new running :class:`~pdart.tasks.Task.Task`
        s to the queue.  None of the tasks may already be in the
        queue, nor may there be duplicates.
        """
        for task in tasks:
            assert task not in self, ('Duplicate task: %s' % task)
        self.running_tasks.insert_tasks(tasks)
        assert self._tasks_are_disjoint()

    def is_running(self, task):
        # type: (Task) -> bool
        """
        Return ``True`` iff the :class:`~pdart.tasks.Task.Task` is
        running in the queue.
        """
        return task in self.running_tasks

    def has_running_tasks(self):
        # type: () -> bool
        """
        Return ``True`` iff there are :class:`~pdart.tasks.Task.Task`
        s running in the queue.
        """
        return bool(self.running_tasks)

    def run_next_task(self):
        # type: () -> Task
        """
        Remove the next pending :class:`~pdart.tasks.Task.Task`, put
        it into
        :attr:`~pdart.tasks.TaskQueue.TaskQueue.running_tasks`, launch
        it, and return it.
        """
        assert self.has_pending_tasks()
        t = self.pending_tasks.popleft()
        assert not self.is_pending(t)
        self.running_tasks.insert_task(t)
        assert self.is_running(t)
        assert self._tasks_are_disjoint()
        self.running_tasks[t].start()
        return t

    def task_finished(self, task):
        # type: (Task) -> None
        """
        Remove the :class:`~pdart.tasks.Task.Task` from
        :attr:`~pdart.tasks.TaskQueue.TaskQueue.running_tasks`.  The
        task must be in
        :attr:`~pdart.tasks.TaskQueue.TaskQueue.running_tasks`.
        """
        assert self.is_running(task)
        del self.running_tasks[task]
        assert self._tasks_are_disjoint()

    def _tasks_are_disjoint(self):
        # type: () -> bool
        """
        Return ``True`` iff there are no
        :class:`~pdart.tasks.Task.Task` that are both pending and
        running.  This is a sanity check.
        """
        return bool(self.running_tasks.isdisjoint(set(self.pending_tasks)))

    def __contains__(self, task):
        """
        Return ``True`` iff the :class:`~pdart.tasks.Task.Task` is
        either pending or running in the queue.
        """
        return task in self.pending_tasks or task in self.running_tasks

    def __nonzero__(self):
        """
        Return ``True`` iff the queue contains any
        :class:`~pdart.tasks.Task.Task` s.
        """
        return bool(self.pending_tasks) or bool(self.running_tasks)

    def __eq__(self, other):
        return self.pending_tasks == other.pending_tasks and \
            self.running_tasks == other.running_tasks
