from pdart.tasks.NullTask import *


def test_NullTask():
    # type: () -> None
    assert NullTask() == NullTask()
    assert NullTask().__hash__() == NullTask().__hash__()


def test_NumberedNullTask():
    # type: () -> None
    nt1 = NumberedNullTask()
    nt2 = NumberedNullTask()
    assert nt1 != nt2
    assert nt1.serial_number + 1 == nt2.serial_number
