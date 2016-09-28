import pdart.tasks.Task
import time


def _a_minute_from_now():
    return time.time() + 60


class NullTask(pdart.tasks.Task.Task):
    """
    A sample Task for testing.  Does nothing.  All instances are
    equal.
    """
    def __init__(self):
        pdart.tasks.Task.Task.__init__(self, _a_minute_from_now())

    def __str__(self):
        return 'NullTask'

    def to_tuple(self):
        return tuple()

    def run(self):
        pass

    def on_success(self, task_runnner):
        pass

    def on_failure(self, task_runnner):
        pass

    def on_termination(self, task_runnner):
        pass

    def on_timeout(self, task_runnner):
        pass


class NumberedNullTask(pdart.tasks.Task.Task):
    """
    A sample Task for testing.  Does nothing.  No two instances are
    equal.
    """
    last_serial_number = 0

    @classmethod
    def get_serial_number(cls):
        res = cls.last_serial_number
        cls.last_serial_number = res + 1
        return res

    def __init__(self):
        pdart.tasks.Task.Task.__init__(self, _a_minute_from_now())
        self.serial_number = NumberedNullTask.get_serial_number()

    def __str__(self):
        return 'NullTask'

    def to_tuple(self):
        return (self.serial_number,)

    def run(self):
        pass

    def on_success(self, task_runnner):
        pass

    def on_failure(self, task_runnner):
        pass

    def on_termination(self, task_runnner):
        pass

    def on_timeout(self, task_runnner):
        pass
