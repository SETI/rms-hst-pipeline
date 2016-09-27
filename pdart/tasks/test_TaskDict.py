from pdart.tasks.TaskDict import *

from pdart.tasks.TestTask import *
import pdart.tasks.TaskProcess


def test_TaskDict():
    td = TaskDict()
    t = NumberedNullTask()
    t2 = NumberedNullTask()
    td.insert_task(t)
    assert t in td
    assert len(td) == 1

    assert t2 not in td
    td.insert_task(t2)
    assert t2 in td
    assert len(td) == 2

    for task in td:
        assert isinstance(td[task], pdart.tasks.TaskProcess.TaskProcess)
