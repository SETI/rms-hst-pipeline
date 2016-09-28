import abc
from pdart.tasks.TaskQueue import *


class Task(object):
    """
    An abstract base class for tasks that can be put into a TaskQueue
    and run.

    They must be objects in order to be pickled when the TaskQueue is
    saved to disk.  They must be hashable to go into the
    TaskQueue.running_tasks set.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, deadline_time):
        assert deadline_time
        self.deadline_time = deadline_time

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def to_tuple(self):
        """
        Return a tuple of subobjects to be used to test equality and
        to hash.
        """
        return None

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return type(self) == type(other) and \
            self.to_tuple() == other.to_tuple()

    @abc.abstractmethod
    def run(self):
        """
        Do the work of the task.  This will only be run in the forked
        process.
        """
        pass

    @abc.abstractmethod
    def on_success(self, task_runnner):
        """Do this in the main process after success of the task."""
        pass

    @abc.abstractmethod
    def on_failure(self, task_runnner):
        """Do this in the main process after failure of the task."""
        pass

    @abc.abstractmethod
    def on_termination(self, task_runnner):
        """Do this in the main process after termination of the task."""
        pass

    @abc.abstractmethod
    def on_timeout(self, task_runnner):
        """Do this in the main process after timeout of the task."""
        pass
