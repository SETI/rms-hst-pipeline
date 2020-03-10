import shutil
import tempfile
import unittest
from fs.path import join
from fs.tempfs import TempFS
from multiversioned.VersionContents import VersionContents
from pdart.pds4.LIDVID import LIDVID
from pdart.pipeline.Utils import *


class Test_Utils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_make_osfs(self):
        with make_osfs(join(self.temp_dir, u"foo")) as fs:
            names = OSFS(self.temp_dir).listdir(u"/")
            self.assertEquals([u"foo"], names)

    def test_make_sv_osfs(self):
        with make_sv_osfs(join(self.temp_dir, u"foo")) as fs:
            names = OSFS(self.temp_dir).listdir(u"/")
            self.assertEquals([u"foo-sv"], names)

    def test_make_mv_osfs(self):
        with make_mv_osfs(join(self.temp_dir, u"foo")) as fs:
            names = OSFS(self.temp_dir).listdir(u"/")
            self.assertEquals([u"foo-mv"], names)

    def test_make_sv_deltas(self):
        with make_sv_osfs(join(self.temp_dir, u"foo")) as base_fs, make_sv_deltas(
            base_fs, join(self.temp_dir, u"bar")
        ) as deltas_fs:
            names = OSFS(self.temp_dir).walk.dirs(u"/")
            self.assertEquals(
                {
                    u"/foo-sv",
                    u"/bar-deltas-sv",
                    u"/bar-deltas-sv/additions",
                    u"/bar-deltas-sv/deletions",
                },
                set(names),
            )

    def test_make_mv_deltas(self):
        with make_mv_osfs(join(self.temp_dir, u"foo")) as base_fs, make_mv_deltas(
            base_fs, join(self.temp_dir, u"bar")
        ) as deltas_fs:
            names = OSFS(self.temp_dir).walk.dirs(u"/")
            self.assertEquals(
                {
                    u"/foo-mv",
                    u"/bar-deltas-mv",
                    u"/bar-deltas-mv/additions",
                    u"/bar-deltas-mv/deletions",
                },
                set(names),
            )

    def test_make_version_view(self):
        with make_mv_osfs(join(self.temp_dir, u"foo")) as base_fs:
            mv = Multiversioned(base_fs)
            lidvid = LIDVID("urn:nasa:pds:b::1.0")
            mv[lidvid] = VersionContents(True, set(), TempFS(), set())
            names = OSFS(self.temp_dir).walk.dirs()
            self.assertEquals(
                {u"/foo-mv", u"/foo-mv/b", u"/foo-mv/b/v$1.0"}, set(names)
            )

            with make_version_view(base_fs, "b") as vv:
                self.assertEquals([u"/b$"], list(vv.walk.dirs()))
