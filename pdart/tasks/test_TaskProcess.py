from TaskProcess import *

import pdart.tasks.TestTask


def test_TaskProcess():
    tp = TaskProcess(pdart.tasks.TestTask.NullTask())
    tp.start()
    tp.join()
    assert tp.status() == 'SUCCEEDED'
