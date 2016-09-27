from pdart.tasks.TestTask import *


def test_TestTask():
    assert TestTask() == TestTask()
    assert TestTask().__hash__() == TestTask().__hash__()


def test_NumberedTestTask():
    nt1 = NumberedTestTask()
    nt2 = NumberedTestTask()
    assert nt1 != nt2
    assert nt1.serial_number + 1 == nt2.serial_number
