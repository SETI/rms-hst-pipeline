from pdart.tasks.RawProcess import *


def _run_task_action(task):
    task.run()


class TaskProcess(TimeoutProcess):
    """A wrapper around RawProcess that runs Task actions"""
    def __init__(self, task):
        assert task
        TimeoutProcess.__init__(self, task.deadline_time,
                                target=_run_task_action, args=(task,))
