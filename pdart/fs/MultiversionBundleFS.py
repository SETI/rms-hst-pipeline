from fs.path import join
from typing import TYPE_CHECKING

from pdart.fs.VersionDirNames import vid_to_dir_name
from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME

if TYPE_CHECKING:
    from pdart.pds4.LIDVID import LIDVID


class MultiversionBundleFS(object):
    def lidvid_to_files_directory(self, lidvid):
        """For a given LIDVID, give the directory that contains its files."""
        # type: (LIDVID) -> unicode
        lid = lidvid.lid()
        vid = lidvid.vid()
        if lidvid.is_bundle_lidvid():
            return join(u'/', lid.bundle_id, vid_to_dir_name(vid))
        elif lidvid.is_collection_lidvid():
            return join(u'/', lid.bundle_id,
                        lid.collection_id, vid_to_dir_name(vid))
        elif lidvid.is_product_lidvid():
            return join(u'/', lid.bundle_id, lid.collection_id,
                        lid.product_id, vid_to_dir_name(vid))
        else:
            assert False, \
                "can't categorize %s as bundle, collection or product" % lid

    def lidvid_to_subdir_versions_path(self, lidvid):
        # type: (LIDVID) -> unicode
        return join(self.lidvid_to_files_directory(lidvid),
                    SUBDIR_VERSIONS_FILENAME)
