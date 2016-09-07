import collections


class TaskQueue(object):
    def __init__(self):
        """Create an empty TaskQueue."""
        self.pending_tasks = collections.deque()
        self.running_tasks = set()
        assert not self

    def append_pending(self, task):
        """
        Add a new pending task to the queue.  The task must not
        already be in the queue.
        """
        assert task not in self, ('Duplicate task: %s' % task)
        self.pending_tasks.append(task)
        assert self._tasks_are_disjoint()

    def extend_pending(self, tasks):
        """
        Add a sequence of new pending tasks to the queue.  None of the
        tasks may already be in the queue, nor may there be
        duplicates.
        """
        for task in tasks:
            assert task not in self, ('Duplicate task: %s' % task)
        self.pending_tasks.extend(tasks)
        assert self._tasks_are_disjoint()

    def is_pending(self, task):
        """Return True iff the task is pending in the queue."""
        return task in self.pending_tasks

    def has_pending_tasks(self):
        """Return True iff there are tasks pending in the queue."""
        return bool(self.pending_tasks)

    def append_running(self, task):
        """
        Add a new running task to the queue.  The task must not
        already be in the queue.
        """
        assert task not in self, ('Duplicate task: %s' % task)
        self.running_tasks.append(task)
        assert self._tasks_are_disjoint()

    def extend_running(self, tasks):
        """
        Add a sequence of new running tasks to the queue.  None of the
        tasks may already be in the queue, nor may there be
        duplicates.
        """
        for task in tasks:
            assert task not in self, ('Duplicate task: %s' % task)
        self.running_tasks.extend(tasks)
        assert self._tasks_are_disjoint()

    def is_running(self, task):
        """Return True iff the task is running in the queue."""
        return task in self.running_tasks

    def has_running_tasks(self):
        """Return True iff there are tasks running in the queue."""
        return bool(self.running_tasks)

    def run_next_task(self):
        """
        Remove the next pending task, put it into the running set,
        launch it, and return it.
        """
        assert self.has_pending_tasks()
        t = self.pending_tasks.popleft()
        assert not self.is_pending(t)
        self.running_tasks.add(t)
        assert self.is_running(t)
        assert self._tasks_are_disjoint()
        self.launch_task(t)
        return t

    def launch_task(self, task):
        """Launch the given task."""
        print ('Launched task %s.' % task)

    def task_finished(self, task):
        """
        Remove the task from the running set.  The task must be in the
        running set.
        """
        assert self.is_running(task)
        self.running_tasks.remove(task)
        assert self._tasks_are_disjoint()

    def _tasks_are_disjoint(self):
        """
        Return True iff there are not task that are both pending and
        running.  This is a sanity check.
        """
        return bool(self.running_tasks.isdisjoint(set(self.pending_tasks)))

    def __contains__(self, task):
        """
        Return True iff the task is either pending or running in the
        queue.
        """
        return task in self.pending_tasks or task in self.running_tasks

    def __nonzero__(self):
        """Return True iff the queue contains any tasks."""
        return bool(self.pending_tasks) or bool(self.running_tasks)
