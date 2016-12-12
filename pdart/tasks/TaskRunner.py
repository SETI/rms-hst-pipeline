"""
An object to queue the tasks to run and keep a given number of them
running simultaneously, launching new tasks when the old ones
complete.
"""
from pdart.tasks.TaskQueue import *

from pdart.tasks.Task import Task  # for mypy

import time


class TaskRunner(object):
    """
    An object that queues up tasks to run and keeps a given number of
    them running simultaneously.  It will queue new tasks when old
    ones have finished.
    """

    _MINUTES = 60
    # type: int
    _DEFAULT_SLEEP_TIME = 10 * _MINUTES
    # type: float

    def __init__(self, capacity):
        # type: (int) -> None
        """
        Create a TaskRunner with given capacity.  Capacity means how
        many processes will run simultaneously.
        """
        assert capacity > 0
        self.capacity = capacity
        self.task_queue = TaskQueue()
        self.SLEEP_TIME = TaskRunner._DEFAULT_SLEEP_TIME

    def run_loop(self):
        # type: () -> None
        """
        Launch up to
        :attr:`~pdart.tasks.TaskRunner.TaskRunner.capacity` tasks in
        external system processes then sleep, waking up every
        :attr:`~pdart.tasks.TaskRunner.TaskRunner.SLEEP_TIME` seconds
        to check on them.  Remove completed tasks, calling their
        post-completion hook methods, launching new tasks if
        available.  Quit when the queue is empty.
        """
        self.fill_running_to_capacity()
        while self.task_queue:
            time.sleep(self.SLEEP_TIME)
            self.prune_by_status()
            # self.check_self_timeout()  # TODO
            self.fill_running_to_capacity()

    def prune_by_status(self):
        # type: () -> None
        """
        Remove tasks that have completed from
        :attr:`~pdart.tasks.TaskRunner.TaskRunner.running_tasks`,
        running their post-completion methods.
        """
        statuses = {}
        for task, process in self.task_queue.running_tasks.iteritems():
            statuses[task] = process.status()
        for task, status in statuses.iteritems():
            assert status != 'INITIALIZED'
            if status == 'SUCCEEDED':
                del self.task_queue.running_tasks[task]
                task.on_success(self)
            elif status == 'FAILED':
                del self.task_queue.running_tasks[task]
                task.on_failure(self)
            elif status == 'TERMINATED':
                del self.task_queue.running_tasks[task]
                task.on_termination(self)
            elif status == 'TIMED_OUT':
                del self.task_queue.running_tasks[task]
                task.on_timeout(self)
            else:
                # Make sure I've caught all the cases.  These all mean
                # the task is still running.
                assert status in ['RUNNING', 'TERMINATING', 'TIMING_OUT']

        # Sanity checking: that only running tasks are still in.  Note
        # that they may have completed in some way since we last
        # called status, so we use the old status values to test.
        for task in self.task_queue.running_tasks:
            # We might have new tasks queued, so check if this is an
            # old task first.
            if task in statuses:
                assert statuses[task] in ['RUNNING',
                                          'TERMINATING',
                                          'TIMING_OUT']

    def extend_pending(self, tasks):
        # type: (List[Task]) -> None
        """
        Add the given tasks to the pending list, then launch some if
        not running at full capacity.
        """
        self.task_queue.extend_pending(tasks)
        self.fill_running_to_capacity()

    def fill_running_to_capacity(self):
        # type: () -> None
        """
        Launch more pending tasks if not already running at full
        capacity.
        """
        tq = self.task_queue
        while tq.has_pending_tasks() and len(tq.running_tasks) < self.capacity:
            tq.run_next_task()
        assert self._at_capacity()

    def _at_capacity(self):
        # type: () -> bool
        """
        Return ``True`` iff the task queue is at maximum capacity:
        i.e., either the number of running tasks equals
        :attr:`~pdart.tasks.TaskRunner.TaskRunner.capacity` or there
        are no more pending tasks to launch.
        """
        N = len(self.task_queue.running_tasks)
        if N < self.capacity:
            return not self.task_queue.pending_tasks
        else:
            return N == self.capacity
