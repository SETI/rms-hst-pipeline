from pdart.tasks.PersistentTaskQueue import *
from pdart.tasks.test_TaskQueue import run_test_TaskQueue


def test_PersistentTaskQueue():
    tq = PersistentTaskQueue('tmp.txt')
    run_test_TaskQueue(tq)
