import shutil
import tempfile
import unittest
from typing import Set

from fs.path import join
from fs.tempfs import TempFS

from pdart.fs.multiversioned.version_contents import VersionContents
from pdart.pds4.lidvid import LIDVID
from pdart.pipeline.utils import *


class Test_utils(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_make_osfs(self) -> None:
        with make_osfs(join(self.temp_dir, "foo")) as fs:
            names = OSFS(self.temp_dir).listdir("/")
            self.assertEqual(["foo"], names)

    def test_make_sv_osfs(self) -> None:
        with make_sv_osfs(join(self.temp_dir, "foo")) as fs:
            names = OSFS(self.temp_dir).listdir("/")
            self.assertEqual(["foo-sv"], names)

    def test_make_mv_osfs(self) -> None:
        with make_mv_osfs(join(self.temp_dir, "foo")) as fs:
            names = OSFS(self.temp_dir).listdir("/")
            self.assertEqual(["foo-mv"], names)

    def test_make_sv_deltas(self) -> None:
        with make_sv_osfs(join(self.temp_dir, "foo")) as base_fs, make_sv_deltas(
            base_fs, join(self.temp_dir, "bar")
        ) as deltas_fs:
            names = OSFS(self.temp_dir).walk.dirs("/")
            self.assertEqual(
                {
                    "/foo-sv",
                    "/bar-deltas-sv",
                    "/bar-deltas-sv/additions",
                    "/bar-deltas-sv/deletions",
                },
                set(names),
            )

    def test_make_mv_deltas(self) -> None:
        with make_mv_osfs(join(self.temp_dir, "foo")) as base_fs, make_mv_deltas(
            base_fs, join(self.temp_dir, "bar")
        ) as deltas_fs:
            names = OSFS(self.temp_dir).walk.dirs("/")
            self.assertEqual(
                {
                    "/foo-mv",
                    "/bar-deltas-mv",
                    "/bar-deltas-mv/additions",
                    "/bar-deltas-mv/deletions",
                },
                set(names),
            )

    def test_make_version_view(self) -> None:
        with make_mv_osfs(join(self.temp_dir, "foo")) as base_fs:
            mv = Multiversioned(base_fs)
            lidvid = LIDVID("urn:nasa:pds:b::1.0")
            no_lidvids: Set[LIDVID] = set()
            mv[lidvid] = VersionContents.create_from_lidvids(
                no_lidvids, TempFS(), set()
            )
            names = OSFS(self.temp_dir).walk.dirs()
            self.assertEqual({"/foo-mv", "/foo-mv/b", "/foo-mv/b/v$1.0"}, set(names))

            with make_version_view(base_fs, "b") as vv:
                self.assertEqual(["/b$"], list(vv.walk.dirs()))
