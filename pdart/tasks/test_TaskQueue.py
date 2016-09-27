from pdart.tasks.TaskQueue import *
from pdart.tasks.Task import *
from pdart.tasks.TestTask import TestTask


def test_TaskQueue():
    tq = TaskQueue()
    assert not tq
    assert not tq.has_pending_tasks()
    assert not tq.has_running_tasks()

    t = TestTask()
    tq.append_pending(t)
    assert tq.has_pending_tasks()
    assert not tq.has_running_tasks()
    assert t in tq
    assert tq.is_pending(t)
    assert not tq.is_running(t)

    rt = tq.run_next_task()
    assert t == rt
    assert tq
    assert not tq.has_pending_tasks()
    assert tq.has_running_tasks()
    assert tq.is_running(rt)

    tq.task_finished(t)
    assert not tq
