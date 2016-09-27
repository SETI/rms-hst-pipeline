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

    def run(self):
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
    def __str__(self):
        pass
