import unittest

from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.UsersFS import *
from pdart.fs.test_FSPrimitives import FSPrimitives_TestBase


class Test_UsersPrimitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        self.base_fs = TempFS()
        self.fs = UsersPrimitives(self.base_fs)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        self.base_fs.close()


class Test_UsersFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return UsersFS(TempFS())
