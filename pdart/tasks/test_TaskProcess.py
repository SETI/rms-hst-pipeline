from TaskProcess import *

import pdart.tasks.NullTask


def test_TaskProcess():
    tp = TaskProcess(pdart.tasks.NullTask.NullTask())
    tp.start()
    tp.join()
    assert tp.status() == 'SUCCEEDED'
