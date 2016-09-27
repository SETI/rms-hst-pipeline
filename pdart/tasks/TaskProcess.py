from pdart.tasks.RawProcess import *


def _run_task_action(task):
    task.run()


class TaskProcess(RawProcess):
    """A wrapper around RawProcess that runs Task actions"""
    def __init__(self, task):
        assert task
        RawProcess.__init__(self, target=_run_task_action, args=(task,))
