from pdart.tasks.TaskRunner import *
from pdart.tasks.TestTask import *


def test_TaskRunner():
    try:
        tr = TaskRunner(-4)
        assert False, "Should have raised exception"
    except:
        pass

    tr = TaskRunner(4)
    assert tr.SLEEP_TIME == 10 * 60
    tr.run_loop()  # exits immediately

    tr = TaskRunner(4)
    tr.SLEEP_TIME = 10.0 / 1000  # wake up every 10ms instead
    tasks = [NumberedNullTask() for i in xrange(0, 50)]

    tr.extend_pending(tasks)
    tr.run_loop()  # exits eventually
