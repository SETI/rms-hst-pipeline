from typing import TYPE_CHECKING
import unittest

from fs.errors import FileExpected
from fs.tempfs import TempFS
from fs.path import join

from pdart.fs.multiversioned.Multiversioned import Multiversioned
from pdart.fs.multiversioned.Utils import dictionary_to_contents
from pdart.fs.multiversioned.VersionContents import VersionContents
from pdart.fs.multiversioned.VersionView import *
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Any, Dict, Set
    from fs.base import FS

############################################################


class Test_VersionView(unittest.TestCase):
    def setUp(self):
        # type: () -> None

        # Build a working bundle containing a collection; both include
        # a set of files, not all on the same level.

        b_lidvid = LIDVID("urn:nasa:pds:b::1.2")
        b_files = {
            "foo.txt": "Hello, world!",
            "counter.txt": "12345",
            "subdir": {"subdir.txt": "xxx"},
        }
        c_lidvid = LIDVID("urn:nasa:pds:b:c::1.1")
        c_files = {
            "undersea.txt": "I'm under c!",
            "deeper": {"leagues40k.txt": "Captain Nemo"},
        }
        self.tempfs = TempFS()
        mv = Multiversioned(self.tempfs)
        mv[b_lidvid] = dictionary_to_contents({c_lidvid}, b_files)
        mv[c_lidvid] = dictionary_to_contents(set(), c_files)

        # Add a second version of the bundle containing nothing, just
        # to check that they stay independent.

        b2_lidvid = LIDVID("urn:nasa:pds:b::2.0")
        b2_files = dict()  # type: Dict[Any, Any]
        mv[b2_lidvid] = dictionary_to_contents(set(), b2_files)

        self.vv = VersionView(mv, b_lidvid)
        self.vv2 = VersionView(mv, b2_lidvid)
        self.mv = mv

    def test_transform_vv_path(self):
        # type: () -> None
        self.assertEqual(u"/", self.vv.transform_vv_path(u"/"))
        with self.assertRaises(ResourceNotFound):
            self.vv.transform_vv_path(u"/foo.txt")
        self.assertEqual(u"/b/v$1.2", self.vv.transform_vv_path(u"/b$"))
        self.assertEqual(u"/b/c/v$1.1", self.vv.transform_vv_path(u"/b$/c$"))

    def test_getinfo(self):
        # type: () -> None
        triples = [
            (u"", True, u"/"),
            (u"b$", True, u"/b$"),
            (u"counter.txt", False, u"/b$/counter.txt"),
            (u"foo.txt", False, u"/b$/foo.txt"),
            (u"subdir", True, u"/b$/subdir"),
            (u"subdir.txt", False, u"/b$/subdir/subdir.txt"),
            (u"c$", True, u"/b$/c$"),
            (u"undersea.txt", False, u"/b$/c$/undersea.txt"),
            (u"deeper", True, u"/b$/c$/deeper"),
            (u"leagues40k.txt", False, u"/b$/c$/deeper/leagues40k.txt"),
        ]
        for triple in triples:
            vv_name, is_dir, vv_path = triple
            info = self.vv.getinfo(vv_path)
            self.assertTrue(is_dir == info.is_dir, str(triple))
            self.assertEqual(vv_name, info.name)

    def test_listdir(self):
        # type: () -> None
        pairs = [
            (u"/", {u"b$"}),
            (u"/b$", {u"counter.txt", u"foo.txt", u"subdir", u"c$"}),
            (u"/b$/subdir", {u"subdir.txt"}),
            (u"/b$/c$", {u"undersea.txt", u"deeper"}),
            (u"/b$/c$/deeper", {u"leagues40k.txt"}),
        ]

        for pair in pairs:
            vv_path, vv_dir_contents = pair
            self.assertEquals(vv_dir_contents, set(self.vv.listdir(vv_path)))

    def test_openbin(self):
        # type: () -> None

        # We'll test through readtext() for part of this instead.

        # Check that the files contain the right values.
        self.assertEqual("12345", self.vv.readtext(u"/b$/counter.txt"))
        self.assertEqual("Hello, world!", self.vv.readtext(u"/b$/foo.txt"))
        self.assertEqual("xxx", self.vv.readtext(u"/b$/subdir/subdir.txt"))
        self.assertEqual("I'm under c!", self.vv.readtext(u"/b$/c$/undersea.txt"))
        self.assertEqual(
            "Captain Nemo", self.vv.readtext(u"/b$/c$/deeper/leagues40k.txt")
        )

        # Try to read and write where files aren't allowed.
        with self.assertRaises(ResourceNotFound):
            self.vv.openbin(u"/foo.txt", "r")
        with self.assertRaises(ResourceReadOnly):
            self.vv.openbin(u"/foo.txt", "w")

        # Try to read a directory.
        with self.assertRaises(FileExpected):
            self.vv.openbin(u"/b$", "r")

        # Try to write or append to the read-only filesystem.
        with self.assertRaises(ResourceReadOnly):
            self.vv.openbin(u"/b$/counter.txt", "w")
        with self.assertRaises(ResourceReadOnly):
            self.vv.openbin(u"/b$/counter.txt", "a")

    def test_getitem(self):
        # type: () -> None

        # A list of VersionViews and LIDVIDs that live inside them.
        vvs_lidvids = [
            (self.vv, LIDVID("urn:nasa:pds:b::1.2")),
            (self.vv, LIDVID("urn:nasa:pds:b:c::1.1")),
            (self.vv2, LIDVID("urn:nasa:pds:b::2.0")),
        ]

        for vv, lidvid in vvs_lidvids:
            # Contents of a LIDVID can be found two different ways.
            # First, they can come directly from the Multiversioned.
            # Second, you can build a VersionView from that LIDVID or
            # from a parent of it, then get the contents from its LID.
            # Check that the two are equal, after reducing LIDVIDs to
            # LIDs.
            lid = lidvid.lid()
            vv_contents = vv[lid]
            mv_contents = self.mv[lidvid].to_lid_version_contents()
            self.assertEqual(mv_contents, vv_contents)

    def test_lid_to_lidvid(self):
        # type: () -> None
        self.assertEqual(
            LIDVID("urn:nasa:pds:b::1.2"), self.vv.lid_to_lidvid(LID("urn:nasa:pds:b"))
        )
        self.assertEqual(
            LIDVID("urn:nasa:pds:b:c::1.1"),
            self.vv.lid_to_lidvid(LID("urn:nasa:pds:b:c")),
        )
        self.assertEqual(
            LIDVID("urn:nasa:pds:b::2.0"), self.vv2.lid_to_lidvid(LID("urn:nasa:pds:b"))
        )

    def test_getsyspath(self):
        self.assertEqual(
            self.tempfs.getsyspath("/b/v$1.2/foo.txt"),
            self.vv.getsyspath(u"b$/foo.txt"),
        )
        self.assertEqual(
            self.tempfs.getsyspath("/b/c/v$1.1/deeper/leagues40k.txt"),
            self.vv.getsyspath(u"b$/c$/deeper/leagues40k.txt"),
        )
