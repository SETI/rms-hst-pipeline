from typing import cast
import unittest

from pdart.rules.ExceptionInfo import ExceptionInfo
from pdart.rules.Result import Failure, Success

_EI = cast(ExceptionInfo, None)


class TestResult(unittest.TestCase):
    def test_is_success(self):
        # type: () -> None
        self.assertTrue(Success(None).is_success())
        self.assertFalse(Failure(_EI).is_success())

    def test_is_failure(self):
        # type: () -> None
        self.assertTrue(Failure(_EI).is_failure())
        self.assertFalse(Success(None).is_failure())
