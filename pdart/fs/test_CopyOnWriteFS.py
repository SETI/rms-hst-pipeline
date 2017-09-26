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

    def test_normalize(self):
        # set the files in the base
        self.cow_base_fs.makedir(u'/foo')
        FOO_BAR_PATH = u'/foo/bar.txt'
        FOO_BAZ_PATH = u'/foo/baz.txt'
        self.cow_base_fs.settext(FOO_BAR_PATH, u'BAR!')
        self.cow_base_fs.settext(FOO_BAZ_PATH, u'BAZ!')

        # lowercase, then re-uppercase BAR.
        self.fs.settext(FOO_BAR_PATH, u'bar!')
        self.assertEqual(u'bar!', self.fs.gettext(FOO_BAR_PATH))
        self.fs.settext(FOO_BAR_PATH, u'BAR!')
        self.assertEqual(u'BAR!', self.fs.gettext(FOO_BAR_PATH))

        # just lowercase BAZ.
        self.fs.settext(FOO_BAZ_PATH, u'baz!')
        self.assertEqual(u'baz!', self.fs.gettext(FOO_BAZ_PATH))

        # sanity test
        delta = self.fs.delta()
        self.assertEqual(set([FOO_BAR_PATH, FOO_BAZ_PATH]), delta.deletions())
        self.assertTrue(delta.additions().exists(FOO_BAR_PATH))
        self.assertTrue(delta.additions().exists(FOO_BAZ_PATH))

        # now normalize.  it should erase the BAR actions.
        self.fs.normalize()

        delta = self.fs.delta()
        self.assertEqual(set([FOO_BAZ_PATH]), delta.deletions())
        self.assertFalse(delta.additions().exists(FOO_BAR_PATH))
        self.assertTrue(delta.additions().exists(FOO_BAZ_PATH))
