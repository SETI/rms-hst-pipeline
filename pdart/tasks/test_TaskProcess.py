from pdart.tasks.TaskProcess import TaskProcess

import pdart.tasks.NullTask


def test_TaskProcess():
    # type: () -> None
    tp = TaskProcess(pdart.tasks.NullTask.NullTask())
    tp.start()
    tp.join()
    assert tp.status() == 'SUCCEEDED'
