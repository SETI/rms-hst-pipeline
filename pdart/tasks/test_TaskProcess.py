from TaskProcess import *

import pdart.tasks.TestTask


def test_TaskProcess():
    tp = TaskProcess(pdart.tasks.TestTask.TestTask())
    tp.start()
    tp.join()
    assert tp.status() == 'SUCCEEDED'
