from pdart.tasks.TaskQueue import *

import time


class TaskRunner(object):
    """
    An object that queues up tasks to run and keeps a given number of
    them running simultaneously.  It will queue new tasks when old
    ones have finished.
    """

    _MINUTES = 60
    _DEFAULT_SLEEP_TIME = 10 * _MINUTES

    def __init__(self, capacity):
        """
        Create a TaskRunner with given capacity.  Capacity means how
        many processes will run simultaneously.
        """
        assert capacity > 0
        self.capacity = capacity
        self.task_queue = TaskQueue()
        self.SLEEP_TIME = TaskRunner._DEFAULT_SLEEP_TIME

    def run_loop(self):
        self.fill_running_to_capacity()
        while self.task_queue:
            time.sleep(self.SLEEP_TIME)
            self.prune_by_status()
            # self.check_self_timeout()  # TODO
            self.fill_running_to_capacity()

    _RUNNING_SET = ['INITIALIZED', 'RUNNING', 'TERMINATING', 'TIMING_OUT']
    _COMPLETED_SET = ['SUCCEEDED', 'FAILED', 'TERMINATED', 'TIMED_OUT']

    def prune_by_status(self):
        """
        Remove tasks that have completed, running their
        post-completion methods (TODO).
        """
        statuses = {}
        for task, process in self.task_queue.running_tasks.iteritems():
            statuses[task] = process.status()
        for task, status in statuses.iteritems():
            if status in TaskRunner._COMPLETED_SET:
                del self.task_queue.running_tasks[task]

    def extend_pending(self, tasks):
        self.task_queue.extend_pending(tasks)
        self.fill_running_to_capacity()

    def fill_running_to_capacity(self):
        tq = self.task_queue
        while tq.has_pending_tasks() and len(tq.running_tasks) < self.capacity:
            tq.run_next_task()
        assert self._at_capacity()

    def _at_capacity(self):
        N = len(self.task_queue.running_tasks)
        if N < self.capacity:
            return not self.task_queue.pending_tasks
        else:
            return N == self.capacity
