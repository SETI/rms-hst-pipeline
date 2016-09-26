class ProcessTask(object):
    def __init__(self, process, task):
        """Create a ProcessTask with the given process and task."""
        assert process
        self.process = process
        assert task
        self.task = task
