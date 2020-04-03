import unittest

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.fs.primitives.DirUtils import (
    dir_to_lid,
    dir_to_lidvid,
    lid_to_dir,
    lidvid_to_dir,
)


class Test_DirUtils(unittest.TestCase):
    def test_lid_to_dir(self) -> None:
        self.assertEqual("/b", lid_to_dir(LID("urn:nasa:pds:b")))
        self.assertEqual("/b/c", lid_to_dir(LID("urn:nasa:pds:b:c")))
        self.assertEqual("/b/c/p", lid_to_dir(LID("urn:nasa:pds:b:c:p")))

    def test_lidvid_to_dir(self) -> None:
        self.assertEqual("/b/v$1.5", lidvid_to_dir(LIDVID("urn:nasa:pds:b::1.5")))
        self.assertEqual("/b/c/v$2.5", lidvid_to_dir(LIDVID("urn:nasa:pds:b:c::2.5")))
        self.assertEqual(
            "/b/c/p/v$333.123", lidvid_to_dir(LIDVID("urn:nasa:pds:b:c:p::333.123"))
        )

    def test_dir_to_lid(self) -> None:
        with self.assertRaises(Exception):
            dir_to_lid("/")
        self.assertEqual(LID("urn:nasa:pds:b"), dir_to_lid("/b"))
        self.assertEqual(LID("urn:nasa:pds:b:c"), dir_to_lid("/b/c"))
        self.assertEqual(LID("urn:nasa:pds:b:c:p"), dir_to_lid("/b/c/p"))
        with self.assertRaises(Exception):
            dir_to_lid("/b/c/p/foo.fits")
        with self.assertRaises(Exception):
            dir_to_lid("/v$1.5")
        with self.assertRaises(Exception):
            dir_to_lid("/b/v$1.5")
        with self.assertRaises(Exception):
            dir_to_lid("/b/c/v$1.5")
        with self.assertRaises(Exception):
            dir_to_lid("/b/c/p/v$1.5")

    def test_dir_to_lidvid(self) -> None:
        with self.assertRaises(Exception):
            dir_to_lidvid("/")
        with self.assertRaises(Exception):
            dir_to_lidvid("/b")
        self.assertEqual(LIDVID("urn:nasa:pds:b::1.5"), dir_to_lidvid("/b/v$1.5"))
        self.assertEqual(LIDVID("urn:nasa:pds:b:c::2.6"), dir_to_lidvid("/b/c/v$2.6"))
        self.assertEqual(
            LIDVID("urn:nasa:pds:b:c:p::333.321"), dir_to_lidvid("/b/c/p/v$333.321")
        )
