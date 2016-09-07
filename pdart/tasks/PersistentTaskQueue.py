from pdart.tasks.TaskQueue import *


class PersistentTaskQueue(TaskQueue):
    def __init__(self, filepath):
        assert filepath is not None
        TaskQueue.__init__(self)
        self.filepath = filepath

    def append_pending(self, task):
        TaskQueue.append_pending(self, task)
        self.write()

    def extend_pending(self, task):
        TaskQueue.extend_pending(self, task)
        self.write()

    def append_running(self, task):
        TaskQueue.append_running(self, task)
        self.write()

    def extend_running(self, task):
        TaskQueue.extend_running(self, task)
        self.write()

    def run_next_task(self):
        t = TaskQueue.run_next_task(self)
        self.write()
        return t

    def task_finished(self, task):
        TaskQueue.task_finished(self, task)
        self.write()

    def write(self):
        # TODO
        pass

    def read(self):
        # TODO
        pass
