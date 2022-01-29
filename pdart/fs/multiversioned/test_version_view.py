import unittest
from typing import Dict

from fs.errors import FileExpected
from fs.tempfs import TempFS

from pdart.fs.multiversioned.multiversioned import Multiversioned
from pdart.fs.multiversioned.test_utils import dictionary_to_contents
from pdart.fs.multiversioned.version_view import *
from pdart.pds4.LIDVID import LIDVID


############################################################


class TestVersionView(unittest.TestCase):
    def setUp(self) -> None:
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
        b2_files: Dict[Any, Any] = dict()
        mv[b2_lidvid] = dictionary_to_contents(set(), b2_files)

        self.vv = VersionView(mv, b_lidvid)
        self.vv2 = VersionView(mv, b2_lidvid)
        self.mv = mv

    def test_transform_vv_path(self) -> None:
        self.assertEqual("/", self.vv.transform_vv_path("/"))
        with self.assertRaises(ResourceNotFound):
            self.vv.transform_vv_path("/foo.txt")
        self.assertEqual("/b/v$1.2", self.vv.transform_vv_path("/b$"))
        self.assertEqual("/b/c/v$1.1", self.vv.transform_vv_path("/b$/c$"))

    def test_getinfo(self) -> None:
        triples = [
            ("", True, "/"),
            ("b$", True, "/b$"),
            ("counter.txt", False, "/b$/counter.txt"),
            ("foo.txt", False, "/b$/foo.txt"),
            ("subdir", True, "/b$/subdir"),
            ("subdir.txt", False, "/b$/subdir/subdir.txt"),
            ("c$", True, "/b$/c$"),
            ("undersea.txt", False, "/b$/c$/undersea.txt"),
            ("deeper", True, "/b$/c$/deeper"),
            ("leagues40k.txt", False, "/b$/c$/deeper/leagues40k.txt"),
        ]
        for triple in triples:
            vv_name, is_dir, vv_path = triple
            info = self.vv.getinfo(vv_path)
            self.assertTrue(is_dir == info.is_dir, str(triple))
            self.assertEqual(vv_name, info.name)

    def test_listdir(self) -> None:
        pairs = [
            ("/", {"b$"}),
            ("/b$", {"counter.txt", "foo.txt", "subdir", "c$"}),
            ("/b$/subdir", {"subdir.txt"}),
            ("/b$/c$", {"undersea.txt", "deeper"}),
            ("/b$/c$/deeper", {"leagues40k.txt"}),
        ]

        for pair in pairs:
            vv_path, vv_dir_contents = pair
            self.assertEqual(vv_dir_contents, set(self.vv.listdir(vv_path)))

    def test_openbin(self) -> None:
        # We'll test through readtext() for part of this instead.

        # Check that the files contain the right values.
        self.assertEqual("12345", self.vv.readtext("/b$/counter.txt"))
        self.assertEqual("Hello, world!", self.vv.readtext("/b$/foo.txt"))
        self.assertEqual("xxx", self.vv.readtext("/b$/subdir/subdir.txt"))
        self.assertEqual("I'm under c!", self.vv.readtext("/b$/c$/undersea.txt"))
        self.assertEqual(
            "Captain Nemo", self.vv.readtext("/b$/c$/deeper/leagues40k.txt")
        )

        # Try to read and write where files aren't allowed.
        with self.assertRaises(ResourceNotFound):
            self.vv.openbin("/foo.txt", "r")
        with self.assertRaises(ResourceReadOnly):
            self.vv.openbin("/foo.txt", "w")

        # Try to read a directory.
        with self.assertRaises(FileExpected):
            self.vv.openbin("/b$", "r")

        # Try to write or append to the read-only filesystem.
        with self.assertRaises(ResourceReadOnly):
            self.vv.openbin("/b$/counter.txt", "w")
        with self.assertRaises(ResourceReadOnly):
            self.vv.openbin("/b$/counter.txt", "a")

    def test_getitem(self) -> None:
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

    def test_lid_to_lidvid(self) -> None:
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

    def test_getsyspath(self) -> None:
        self.assertEqual(
            self.tempfs.getsyspath("/b/v$1.2/foo.txt"),
            self.vv.getsyspath("b$/foo.txt"),
        )
        self.assertEqual(
            self.tempfs.getsyspath("/b/c/v$1.1/deeper/leagues40k.txt"),
            self.vv.getsyspath("b$/c$/deeper/leagues40k.txt"),
        )
