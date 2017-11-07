from fs.path import iteratepath, join
from fs.wrap import WrapFS
from typing import TYPE_CHECKING

from pdart.fs.SubdirVersions import read_subdir_versions_from_directory, \
    read_subdir_versions_from_path, write_subdir_versions_to_path
from pdart.fs.DirUtils import _dir_part_to_vid, _is_dir_part, lid_to_dir, \
    lidvid_to_dir, _vid_to_dir_part
from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from typing import Dict, List, Tuple
    from pdart.pds4.LIDVID import LIDVID
    from pdart.pds4.LID import LID


class MultiversionBundleFS(WrapFS):
    """
    Adds some utility functions to a fs.base.FS and denotes that it contains
    multiple versions of bundles in our format.  TODO Specify the format.
    """

    def __init__(self, wrap_fs):
        WrapFS.__init__(self, wrap_fs)

    def make_lidvid_directories(self, lidvid):
        """
        Create the contents directory and create an empty
        subdir_versions file if none exists for this LID version.
        """
        # type: (LIDVID) -> None
        path = lidvid_to_dir(lidvid)
        self._wrap_fs.makedirs(path, recreate=True)
        dict_path = lidvid_to_subdir_versions_path(lidvid)
        is_product = lidvid.lid().product_id is not None
        if not is_product and not self._wrap_fs.exists(dict_path):
            write_subdir_versions_to_path(self, dict_path, {})

    def make_lid_directories(self, lid):
        """
        Create the directory for this LID.
        """
        # type: (LID) -> None
        path = lid_to_dir(lid)
        self._wrap_fs.makedirs(path, recreate=True)

    def read_lidvid_subdir_versions(self, lidvid):
        """
        For a given LIDVID, read and return the dictionary stored
        in the subdir_versions file.
        """
        # type: (LIDVID) -> Dict[unicode, unicode]
        path = lidvid_to_subdir_versions_path(lidvid)
        return read_subdir_versions_from_path(self, path)

    def write_lidvid_subdir_versions(self, lidvid, d):
        """
        For a given LIDVID, write a dictionary into the subdir_versions file.
        """
        # type: (LIDVID, Dict[unicode, unicode]) -> None
        path = lidvid_to_subdir_versions_path(lidvid)
        return write_subdir_versions_to_path(self, path, d)

    def read_subdirectory_paths(self, lidvid):
        """
        Read the dictionary that maps from subdirectories
        to their paths.
        """
        # type: (LIDVID) -> Dict[unicode, unicode]
        d = self.read_lidvid_subdir_versions(lidvid)
        lid = lidvid.lid()
        base_dir = lid_to_dir(lid)

        def make_version_subdir(dir_name, version):
            return join(base_dir, dir_name, _vid_to_dir_part(VID(version)))

        return {dir_name: make_version_subdir(dir_name, version)
                for dir_name, version in d.items()}

    def add_subcomponent(self, parent_lidvid, child_lidvid):
        """
        Add directories for the child component to the parent
        component, including the subdir_versions dictionary.
        """
        # type: (LIDVID, LIDVID) -> None
        child_lid = child_lidvid.lid()
        assert child_lid.parent_lid() == parent_lidvid.lid(), \
            '%s is not parent to %s' % (parent_lidvid, child_lidvid)
        d = self.read_lidvid_subdir_versions(parent_lidvid)
        last_id = child_lid.product_id or child_lid.collection_id
        d[last_id] = str(child_lidvid.vid())
        self.write_lidvid_subdir_versions(parent_lidvid, d)
        self.make_lidvid_directories(child_lidvid)

    def read_subcomponent_lidvids(self, lidvid):
        """
        Read the LIDVIDs of the subcomponents of this LIDVID.
        """
        # type: (LIDVID) -> Dict[unicode, LIDVID]
        d = self.read_lidvid_subdir_versions(lidvid)
        lid = lidvid.lid()

        def make_subcomp_lidvid(dir_name, version):
            # type: (unicode, unicode) -> LIDVID
            assert False, 'unimplemented'
            pass

        return {dir_name: make_subcomp_lidvid(dir_name, version)
                for dir_name, version in d.items()}

    def directory_contents(self, dir_path):
        # type: (unicode) -> Tuple[Dict[unicode, unicode], List[unicode]]
        files = [info.name
                 for info in self.scandir(dir_path)
                 if info.is_file and info.name != SUBDIR_VERSIONS_FILENAME]
        if len(iteratepath(dir_path)) >= 4:
            # if we're a versioned product dir: /b/c/p/v$n
            d = {}
            # type: Dict[unicode, unicode]
        else:
            d = read_subdir_versions_from_directory(self, dir_path)
        return d, files

    def current_vid(self, lid):
        """
        For a given LID, find the current (latest) VID in the filesystem.
        """
        # type: (LID) -> VID
        path = lid_to_dir(lid)
        vids = [_dir_part_to_vid(dir_name)
                for dir_name in self.listdir(path)
                if _is_dir_part(dir_name)]
        if not vids:
            return VID('0')
        else:
            return max(vids)

    def validate(self):
        # type: () -> None
        """
        Validate that the filesystem follows the rules for a
        MultiversionBundleFS.
        """
        pass

    def current_bundle_lidvid(self):
        # type: () -> LIDVID
        root_contents = self.listdir(u'/')
        assert len(root_contents) == 1
        bundle_id = root_contents[0]
        lid = LID('urn:nasa:pds:' + bundle_id)
        vid = self.current_vid(lid)
        return LIDVID.create_from_lid_and_vid(lid, vid)


def lidvid_to_subdir_versions_path(lidvid):
    """
    For a given LIDVID, give the path to the subdir_versions file
    indicating its subdirectories.
    """
    # type: (LIDVID) -> unicode
    return join(lidvid_to_dir(lidvid),
                SUBDIR_VERSIONS_FILENAME)
