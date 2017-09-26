import unittest

from fs.test import FSTestCases

from CopyOnWriteFS import *


class TestCopyOnWriteFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        self.cow_base_fs = TempFS()
        self.cow_delta_fs = TempFS()
        return CopyOnWriteFS(self.cow_base_fs, self.cow_delta_fs)

    def test_fs_delta(self):
        delta = self.fs.delta()
        self.assertTrue(delta)
        self.assertIs(self.cow_delta_fs, delta.additions())
        self.assertFalse(delta.deletions())

        BAR_PATH = u'/bar.txt'
        FOO_PATH = u'/foo.txt'

        self.cow_base_fs.settext(FOO_PATH, u'FOO!')
        self.fs.remove(FOO_PATH)
        delta = self.fs.delta()
        self.assertTrue(self.cow_base_fs.exists(FOO_PATH))
        self.assertFalse(delta.additions().exists(FOO_PATH))
        self.assertEqual(set([FOO_PATH]), delta.deletions())

        self.fs.settext(BAR_PATH, u'BAR!')
        delta = self.fs.delta()
        self.assertFalse(self.cow_base_fs.exists(BAR_PATH))
        self.assertTrue(delta.additions().exists(BAR_PATH))
