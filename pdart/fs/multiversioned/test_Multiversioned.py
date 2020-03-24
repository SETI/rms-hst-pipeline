import unittest

from fs.memoryfs import MemoryFS

from pdart.fs.cowfs.COWFS import COWFS
from pdart.fs.multiversioned.Multiversioned import *
from pdart.fs.multiversioned.Utils import dictionary_to_contents
from pdart.fs.multiversioned.VersionView import VersionView


def is_new(lidvid: LIDVID, contents: VersionContents, mv: Multiversioned) -> bool:
    return mv[lidvid] != contents


class Test_Multiversioned(unittest.TestCase):
    def test_lid_path(self) -> None:
        self.assertEqual("/b", lid_path(LID("urn:nasa:pds:b")))
        self.assertEqual("/b/c", lid_path(LID("urn:nasa:pds:b:c")))
        self.assertEqual("/b/c/p", lid_path(LID("urn:nasa:pds:b:c:p")))

    def test_lidvid_path(self) -> None:
        self.assertEqual("/b/v$1.5", lidvid_path(LIDVID("urn:nasa:pds:b::1.5")))
        self.assertEqual("/b/c/v$2.5", lidvid_path(LIDVID("urn:nasa:pds:b:c::2.5")))
        self.assertEqual(
            "/b/c/p/v$333.123", lidvid_path(LIDVID("urn:nasa:pds:b:c:p::333.123"))
        )

    def test_iter_and_len(self) -> None:
        mv = Multiversioned(MemoryFS())
        self.assertEqual(0, len(mv))
        lidvids = [
            LIDVID("urn:nasa:pds:b::1.5"),
            LIDVID("urn:nasa:pds:b:c::2.5"),
            LIDVID("urn:nasa:pds:b:c:p::333.123"),
        ]

        for i, lidvid in enumerate(lidvids):
            self.assertEqual(i, len(mv))
            mv.make_lidvid_dir(lidvid)
            self.assertTrue(lidvid in mv)

    def test_get_set_items(self) -> None:
        no_lidvids: Set[LIDVID] = set()

        # Set empty contents
        mv = Multiversioned(MemoryFS())
        empty_lidvid = LIDVID("urn:nasa:pds:empty-bundle::3.14")
        empty_contents = VersionContents(True, no_lidvids, MemoryFS(), set())
        mv[empty_lidvid] = empty_contents
        self.assertTrue(empty_lidvid in mv)
        self.assertEqual(1, len(mv))
        self.assertEqual(empty_contents, mv[empty_lidvid])

        # TODO Is IndexError the right exception to raise?
        with self.assertRaises(IndexError):
            mv[empty_lidvid] = empty_contents

        # Set contents with a single file down a long path
        single_file_lidvid = LIDVID("urn:nasa:pds:single-file::3.14")
        single_file_fs = MemoryFS()
        single_file_path = "/down/a/lot/of/dirs/text.txt"
        single_file_fs.makedirs(fs.path.dirname(single_file_path), None, True)
        single_file_fs.writetext(single_file_path, "Hello, there!")
        single_file_contents = VersionContents(
            True, no_lidvids, single_file_fs, set([single_file_path])
        )
        mv[single_file_lidvid] = single_file_contents
        self.assertTrue(empty_lidvid in mv)
        self.assertTrue(single_file_lidvid in mv)
        print("****", set(mv.lidvids()))
        self.assertEqual(2, len(mv))
        self.assertEqual(single_file_contents, mv[single_file_lidvid])

        # Test that LIDVIDs get put correctly into the
        # subdir$version.txts.
        hierarchic = Multiversioned(MemoryFS())
        b_lidvid = LIDVID("urn:nasa:pds:b::1.5")
        c_lidvid = LIDVID("urn:nasa:pds:b:c::2.5")
        p_lidvid = LIDVID("urn:nasa:pds:b:c:p::333.123")

        p_contents = VersionContents(True, no_lidvids, MemoryFS(), set([]))
        hierarchic[p_lidvid] = p_contents
        self.assertEqual(p_contents, hierarchic[p_lidvid])

        c_contents = VersionContents(True, set([p_lidvid]), MemoryFS(), set([]))
        hierarchic[c_lidvid] = c_contents
        self.assertEqual(c_contents, hierarchic[c_lidvid])

        b_contents = VersionContents(True, set([c_lidvid]), MemoryFS(), set([]))
        hierarchic[b_lidvid] = b_contents
        self.assertEqual(b_contents, hierarchic[b_lidvid])

        self.assertEqual(3, len(hierarchic))
        self.assertEqual({p_lidvid, c_lidvid, b_lidvid}, hierarchic.lidvids())

    def test_update_from_single_version(self) -> None:
        fs = MemoryFS()
        mv = Multiversioned(fs)

        d = {
            "file1.txt": "file1",
            "file2.txt": "file2",
            "dir1": {"file1.txt": "file1", "file2.txt": "file2"},
            "dir2": {"file1.txt": "file1", "file2.txt": "file2"},
        }

        bundle_lidvid = LIDVID("urn:nasa:pds:b::1.0")
        bundle_lid = bundle_lidvid.lid()

        no_lidvids: Set[LIDVID] = set()

        def create_bundle() -> None:
            lidvids = [create_collection(bundle_lid, c) for c in ["c1", "c2"]]
            contents = dictionary_to_contents(set(lidvids), d)
            mv.add_contents_if(is_new, bundle_lid, contents)

        def create_collection(bundle_lid: LID, c: str) -> LIDVID:
            lid = bundle_lid.extend_lid(c)
            lidvids = [create_product(lid, p) for p in ["p1", "p2"]]
            contents = dictionary_to_contents(set(lidvids), d)
            return mv.add_contents_if(is_new, lid, contents)

        def create_product(coll_lid: LID, p: str) -> LIDVID:
            lid = coll_lid.extend_lid(p)

            contents = dictionary_to_contents(no_lidvids, d)
            return mv.add_contents_if(is_new, lid, contents)

        create_bundle()

        vv = VersionView(mv, bundle_lidvid)
        c = COWFS(vv, MemoryFS(), MemoryFS())

        path = "/b$/c2$/p1$/dir1/file2.txt"
        c.writetext(path, "xxxx")

        latest_lidvid = mv.latest_lidvid(LID("urn:nasa:pds:b"))
        # Update from the COWFS.
        mv.update_from_single_version(is_new, c)
        self.assertNotEqual(latest_lidvid, mv.latest_lidvid(LID("urn:nasa:pds:b")))
        latest_lidvid = mv.latest_lidvid(LID("urn:nasa:pds:b"))

        # changed files are changed
        self.assertEqual("file2", fs.readtext("b/c2/p1/v$1.0/dir1/file2.txt"))
        self.assertEqual("xxxx", fs.readtext("b/c2/p1/v$2.0/dir1/file2.txt"))

        # unchanged files are unchanged
        self.assertEqual("file1", fs.readtext("b/c2/p1/v$1.0/dir1/file1.txt"))
        self.assertEqual("file1", fs.readtext("b/c2/p1/v$2.0/dir1/file1.txt"))

        # Change started in b/c2/p1.  Check which versions are affected.

        self.assertEqual(
            VID("2.0"), cast(LIDVID, mv.latest_lidvid(LID("urn:nasa:pds:b"))).vid()
        )

        self.assertEqual(
            VID("1.0"), cast(LIDVID, mv.latest_lidvid(LID("urn:nasa:pds:b:c1"))).vid()
        )
        self.assertEqual(
            VID("2.0"), cast(LIDVID, mv.latest_lidvid(LID("urn:nasa:pds:b:c2"))).vid()
        )

        self.assertEqual(
            VID("2.0"),
            cast(LIDVID, mv.latest_lidvid(LID("urn:nasa:pds:b:c2:p1"))).vid(),
        )
        self.assertEqual(
            VID("1.0"),
            cast(LIDVID, mv.latest_lidvid(LID("urn:nasa:pds:b:c2:p2"))).vid(),
        )

        # Now try updating again.  Nothing should change.
        mv.update_from_single_version(is_new, c)
        self.assertEqual(latest_lidvid, mv.latest_lidvid(LID("urn:nasa:pds:b")))
