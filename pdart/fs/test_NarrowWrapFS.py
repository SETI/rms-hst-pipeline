import unittest
from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.NarrowWrapFS import *


class TestNarrowWrapFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return NarrowWrapFS(TempFS())
