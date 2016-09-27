import pdart.tasks.TaskProcess


class TaskDict(dict):
    """
    A Python dictionary with Task keys and TaskProcess values and
    functionality to insert keys automatically making the
    corresponding value.
    """
    def __init__(self):
        pass

    def insert_task(self, task):
        """
        Insert a new task to run into the TaskDict.  The task must not
        already be in the TaskDict.
        """
        assert task
        assert task not in self, ('Duplicate task: %s' % task)
        self[task] = pdart.tasks.TaskProcess.TaskProcess(task)

    def insert_tasks(self, tasks):
        """
        Insert new tasks to run into the TaskDict.  The tasks must not
        already be in the TaskDict.
        """
        assert tasks
        for task in tasks:
            self.insert_task(task)

    def isdisjoint(self, other_tasks):
        """
        Return True if the set of keys for this dictionary is disjoint
        from other_tasks.
        """
        for task in other_tasks:
            if task in self:
                return False
        return True
