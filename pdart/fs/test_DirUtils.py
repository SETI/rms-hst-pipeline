import unittest

from pdart.fs.DirUtils import *


class test_DirUtils(unittest.TestCase):
    def test_lid_to_dir(self):
        # type: () -> None
        self.assertEqual(u'/b', lid_to_dir(LID('urn:nasa:pds:b')))
        self.assertEqual(u'/b/c', lid_to_dir(LID('urn:nasa:pds:b:c')))
        self.assertEqual(u'/b/c/p', lid_to_dir(LID('urn:nasa:pds:b:c:p')))

    def test_lidvid_to_dir(self):
        # type: () -> None
        self.assertEqual(u'/b/v$1.5',
                         lidvid_to_dir(LIDVID('urn:nasa:pds:b::1.5')))
        self.assertEqual(u'/b/c/v$2',
                         lidvid_to_dir(LIDVID('urn:nasa:pds:b:c::2')))
        self.assertEqual(u'/b/c/p/v$333',
                         lidvid_to_dir(LIDVID('urn:nasa:pds:b:c:p::333')))

    def test_dir_to_lid(self):
        # type: () -> None
        with self.assertRaises(Exception):
            dir_to_lid(u'/')
        self.assertEqual(LID('urn:nasa:pds:b'), dir_to_lid(u'/b'))
        self.assertEqual(LID('urn:nasa:pds:b:c'), dir_to_lid(u'/b/c'))
        self.assertEqual(LID('urn:nasa:pds:b:c:p'), dir_to_lid(u'/b/c/p'))
        with self.assertRaises(Exception):
            dir_to_lid(u'/b/c/p/foo.fits')
        with self.assertRaises(Exception):
            dir_to_lid(u'/v$1')
        with self.assertRaises(Exception):
            dir_to_lid(u'/b/v$1')
        with self.assertRaises(Exception):
            dir_to_lid(u'/b/c/v$1')
        with self.assertRaises(Exception):
            dir_to_lid(u'/b/c/p/v$1')

    def test_dir_to_lidvid(self):
        # type: () -> None
        with self.assertRaises(Exception):
            dir_to_lidvid(u'/')
        with self.assertRaises(Exception):
            dir_to_lidvid(u'/b')
        self.assertEqual(LIDVID('urn:nasa:pds:b::1.5'),
                         dir_to_lidvid(u'/b/v$1.5'))
        self.assertEqual(LIDVID('urn:nasa:pds:b:c::2'),
                         dir_to_lidvid(u'/b/c/v$2'))
        self.assertEqual(LIDVID('urn:nasa:pds:b:c:p::333'),
                         dir_to_lidvid(u'/b/c/p/v$333'))
