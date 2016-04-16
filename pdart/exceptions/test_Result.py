import unittest

from pdart.exceptions.Result import *


class TestResult(unittest.TestCase):
    def test_is_success(self):
        self.assertTrue(Success(None).is_success())
        self.assertFalse(Failure([]).is_success())

    def test_is_failure(self):
        self.assertTrue(Failure([]).is_failure())
        self.assertFalse(Success(None).is_failure())
