"""
Represenation for tasks to be run.
"""
import abc
from pdart.tasks.TaskQueue import *


class Task(object):
    """
    An abstract base class for tasks that can be put into a
    :class:`~pdart.tasks.TaskQueue.TaskQueue` and run.

    They must be picklable for when the
    :class:`~pdart.tasks.TaskQueue.TaskQueue` is saved to disk.  They
    must be hashable to go into the
    :class:`~pdart.tasks.TaskQueue.TaskQueue`'s
    :attr:`~pdart.tasks.TaskQueue.TaskQueue.running_tasks` dictionary.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, deadline_time):
        """
        Create a :class:`~pdart.tasks.Task.Task` with the given
        deadline time.  If it's still running past this time, it is
        liable to be interrupted by the
        :class:`~pdart.tasks.TaskRunner.TaskRunner`.
        """
        assert deadline_time
        self.deadline_time = deadline_time

    @abc.abstractmethod
    def __str__(self):
        """
        Return a string describing the object.  Required for
        debugging.
        """
        pass

    @abc.abstractmethod
    def to_tuple(self):
        """
        Return a tuple of subobjects to be used to test equality and
        to hash.
        """
        return None

    def __hash__(self):
        """
        Return an integer hashcode for the object.  Required since
        tasks are used as keys in dictionaries.
        """
        return hash(self.to_tuple())

    def __eq__(self, other):
        """
        Return true if the other object is a task of the same type and
        gives equal results for
        :method:`pdart.tasks.Task.Task.to_tuple`.  NOTE: the types
        must be exactly the same, so subtyping will create unequal
        objects.  Also, note that we're implementing *value equality*
        not *pointer equality* so that if we write a task to disk and
        then read it back, the new task and the original will be
        equal, although they live in different places in memory.  This
        implies that each task of a given type must have a unique
        `to_tuple()` value.
        """
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
