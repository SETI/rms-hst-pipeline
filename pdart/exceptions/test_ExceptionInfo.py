from pdart.exceptions.ExceptionInfo import *


def test_ExceptionInfo():
    try:
        ce = CalculationException('<msg>', Exception('foo'))
        self.assertFalse(True)
    except AssertionError:
        pass
