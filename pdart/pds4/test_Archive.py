import shutil
import tempfile
import unittest

from pdart.pds4.Archive import Archive


class TestArchive(unittest.TestCase):
    def test_init(self):
        # type: () -> None
        with self.assertRaises(Exception):
            Archive(None)
        with self.assertRaises(Exception):
            Archive("I'm/betting/this/directory/doesn't/exist")

        Archive('/')  # guaranteed to exist

        # but try with another directory
        tempdir = tempfile.mkdtemp()
        try:
            Archive(tempdir)
        finally:
            shutil.rmtree(tempdir)

    def test_str(self):
        # type: () -> None
        tempdir = tempfile.mkdtemp()
        a = Archive(tempdir)
        self.assertEqual(repr(tempdir), str(a))

    def test_eq(self):
        # type: () -> None
        tempdir = tempfile.mkdtemp()
        tempdir2 = tempfile.mkdtemp()
        self.assertEquals(Archive(tempdir), Archive(tempdir))
        self.assertNotEquals(Archive(tempdir), Archive(tempdir2))

    def test_repr(self):
        # type: () -> None
        tempdir = tempfile.mkdtemp()
        a = Archive(tempdir)
        self.assertEqual('Archive(%r)' % tempdir, repr(a))

    def test_is_valid_instrument(self):
        # type: () -> None
        self.assertTrue(Archive.is_valid_instrument('wfc3'))
        self.assertTrue(Archive.is_valid_instrument('wfpc2'))
        self.assertTrue(Archive.is_valid_instrument('acs'))
        self.assertFalse(Archive.is_valid_instrument('Acs'))
        self.assertFalse(Archive.is_valid_instrument('ACS'))
        self.assertFalse(Archive.is_valid_instrument('ABC'))
        self.assertFalse(Archive.is_valid_instrument(None))

    def test_is_valid_proposal(self):
        # type: () -> None
        self.assertFalse(Archive.is_valid_proposal(-1))
        self.assertTrue(Archive.is_valid_proposal(0))
        self.assertTrue(Archive.is_valid_proposal(1))
        self.assertFalse(Archive.is_valid_proposal(100000))
        # self.assertFalse(Archive.is_valid_proposal(3.14159265))
        # self.assertFalse(Archive.is_valid_proposal('xxx'))
        self.assertFalse(Archive.is_valid_proposal(None))

    def test_is_valid_visit(self):
        # type: () -> None
        self.assertTrue(Archive.is_valid_visit('01'))
        self.assertFalse(Archive.is_valid_visit('xxx'))
        # self.assertFalse(Archive.is_valid_visit(01))
        self.assertFalse(Archive.is_valid_visit(None))
