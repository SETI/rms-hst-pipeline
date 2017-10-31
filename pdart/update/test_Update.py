import unittest

from fs.path import join
from fs.tempfs import TempFS

from pdart.fs.LidToDirName import lid_to_dir_name
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView
from pdart.pds4.LIDVID import LIDVID
from pdart.update.Update import update_bundle


class Test_Update(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        mvfs = MultiversionBundleFS(TempFS())

        # >>> write empty_tree
        bundle_lidvid = LIDVID('urn:nasa:pds:b::1')
        collection_lidvid = LIDVID('urn:nasa:pds:b:c::1')
        product_lidvid = LIDVID('urn:nasa:pds:b:c:p::1')

        mvfs.make_lidvid_directories(bundle_lidvid)
        mvfs.add_subcomponent(bundle_lidvid, collection_lidvid)
        mvfs.add_subcomponent(collection_lidvid, product_lidvid)
        # <<< end write empty tree

        vv = VersionView(bundle_lidvid, mvfs)
        vv.touch(join(lid_to_dir_name(product_lidvid.lid()), "foo.fits"))

        self.multiversioned_fs = mvfs

        self.last_bundle_lidvid = bundle_lidvid

    def test_update(self):
        # type: () -> None
        def update(cow_fs):
            # type: (CopyOnWriteFS) -> None
            pass

        update_bundle(self.multiversioned_fs,
                      self.last_bundle_lidvid,
                      True, update)
        # doing nothing does not update the bundle
        self.assertEqual(self.last_bundle_lidvid,
                         self.multiversioned_fs.current_bundle_lidvid())
