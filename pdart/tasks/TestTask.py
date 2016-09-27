import pdart.tasks.Task


class TestTask(pdart.tasks.Task.Task):
    """
    A sample Task for testing.  All instances are equal.
    """
    def __str__(self):
        return 'TestTask'

    def to_tuple(self):
        return tuple()


class NumberedTestTask(pdart.tasks.Task.Task):
    """
    A sample Task for testing.  No two instances are equal.
    """
    last_serial_number = 0

    @classmethod
    def get_serial_number(cls):
        res = cls.last_serial_number
        cls.last_serial_number = res + 1
        return res

    def __init__(self):
        self.serial_number = NumberedTestTask.get_serial_number()

    def __str__(self):
        return 'TestTask'

    def to_tuple(self):
        return (self.serial_number,)
