import unittest

from pdart.rules.Result import Failure, Success


class TestResult(unittest.TestCase):
    def test_is_success(self):
        # type: () -> None
        self.assertTrue(Success(None).is_success())
        self.assertFalse(Failure([]).is_success())

    def test_is_failure(self):
        # type: () -> None
        self.assertTrue(Failure([]).is_failure())
        self.assertFalse(Success(None).is_failure())
