from pdart.tasks.TestTask import *


def test_NullTask():
    assert NullTask() == NullTask()
    assert NullTask().__hash__() == NullTask().__hash__()


def test_NumberedNullTask():
    nt1 = NumberedNullTask()
    nt2 = NumberedNullTask()
    assert nt1 != nt2
    assert nt1.serial_number + 1 == nt2.serial_number
