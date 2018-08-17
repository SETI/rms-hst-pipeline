import unittest

from pdart.fs.DeletionSet import DeletionSet


class TestDeletionSet(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self._deletion_set = DeletionSet()

    def test_init(self):
        # type: () -> None
        self.assertFalse(self._deletion_set.as_set())

    def test_delete(self):
        # type: () -> None
        FOO = '/foo/bar/baz'
        self.assertFalse(self._deletion_set.is_deleted(FOO))
        self._deletion_set.delete(FOO)
        self.assertTrue(self._deletion_set.is_deleted(FOO))
        self.assertTrue(FOO in self._deletion_set.as_set())

    def test_undelete(self):
        # type: () -> None
        FOO = '/foo/bar/baz'
        self._deletion_set.delete(FOO)
        self._deletion_set.undelete(FOO)
        self.assertFalse(self._deletion_set.is_deleted(FOO))
        self.assertTrue(FOO not in self._deletion_set.as_set())

    def test_as_set(self):
        # type: () -> None
        FOO = '/foo'
        BAR = '/bar'
        self.assertFalse(self._deletion_set.as_set())
        self._deletion_set.delete(FOO)
        self.assertEquals(set([FOO]), self._deletion_set.as_set())
        self._deletion_set.delete(BAR)
        self.assertEquals(set([FOO, BAR]), self._deletion_set.as_set())
        self._deletion_set.undelete(FOO)
        self.assertEquals(set([BAR]), self._deletion_set.as_set())
