import unittest

from SetDeletionPredicate import *


class TestSetDeletionPredicate(unittest.TestCase):
    def setUp(self):
        self._deletion_predicate = SetDeletionPredicate()

    def test_init(self):
        self.assertFalse(self._deletion_predicate.as_set())

    def test_delete(self):
        FOO = '/foo/bar/baz'
        self.assertFalse(self._deletion_predicate.is_deleted(FOO))
        self._deletion_predicate.delete(FOO)
        self.assertTrue(self._deletion_predicate.is_deleted(FOO))
        self.assertTrue(FOO in self._deletion_predicate.as_set())

    def test_undelete(self):
        FOO = '/foo/bar/baz'
        self._deletion_predicate.delete(FOO)
        self._deletion_predicate.undelete(FOO)
        self.assertFalse(self._deletion_predicate.is_deleted(FOO))
        self.assertTrue(FOO not in self._deletion_predicate.as_set())

    def test_as_set(self):
        FOO = '/foo'
        BAR = '/bar'
        self.assertFalse(self._deletion_predicate.as_set())
        self._deletion_predicate.delete(FOO)
        self.assertEquals(set([FOO]), self._deletion_predicate.as_set())
        self._deletion_predicate.delete(BAR)
        self.assertEquals(set([FOO, BAR]), self._deletion_predicate.as_set())
        self._deletion_predicate.undelete(FOO)
        self.assertEquals(set([BAR]), self._deletion_predicate.as_set())
