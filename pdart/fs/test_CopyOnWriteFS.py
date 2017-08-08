import unittest
from fs.tempfs import TempFS
from fs.test import FSTestCases

from CopyOnWriteFS import *


class TestCopyOnWriteFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return CopyOnWriteFS(TempFS())
