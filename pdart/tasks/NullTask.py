"""
:class:`~pdart.tasks.Task.Task` subclasses used for testing.
"""
import pdart.tasks.Task
import time


def _a_minute_from_now():
    return time.time() + 60


class NullTask(pdart.tasks.Task.Task):
    """
    A sample :class:`~pdart.tasks.Task.Task` for testing which times
    out in a minute and does nothing.  All instances are equal.
    """
    def __init__(self):
        # type () -> None
        pdart.tasks.Task.Task.__init__(self, _a_minute_from_now())

    def __str__(self):
        return 'NullTask'

    def to_tuple(self):
        return tuple()

    def run(self):
        pass

    def on_success(self, target):
        pass

    def on_failure(self, target):
        pass

    def on_termination(self, target):
        pass

    def on_timeout(self, target):
        pass


class NumberedNullTask(pdart.tasks.Task.Task):
    """
    A sample :class:`~pdart.tasks.Task.Task` for testing, which times
    out after a minute and does nothing.  Each instance is tagged with
    a unique serial number so no two instances are equal.
    """
    last_serial_number = 0

    @classmethod
    def get_serial_number(cls):
        # type: () -> int
        """Return a new serial number."""
        res = cls.last_serial_number
        cls.last_serial_number = res + 1
        return res

    def __init__(self):
        pdart.tasks.Task.Task.__init__(self, _a_minute_from_now())
        self.serial_number = NumberedNullTask.get_serial_number()

    def __str__(self):
        return ('NumberedNullTask(%d)' % self.serial_number)

    def to_tuple(self):
        return (self.serial_number,)

    def run(self):
        pass

    def on_success(self, target):
        pass

    def on_failure(self, target):
        pass

    def on_termination(self, target):
        pass

    def on_timeout(self, target):
        pass
