from fs.errors import DirectoryExpected, FileExpected, ResourceNotFound
from fs.info import Info
from fs.path import abspath, basename, dirname, iteratepath, normpath

from pdart.fs.MultiversionBundleFS \
    import MultiversionBundleFS, lidvid_to_contents_directory_path
from pdart.fs.ReadOnlyView import ReadOnlyView
from pdart.fs.SubdirVersions import *
from pdart.fs.VersionDirNames import version_id_to_dir_name, vid_to_dir_name
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from typing import Dict, Tuple
    from fs.osfs import OSFS


def _make_raw_dir_info(name):
    # type: (unicode) -> Dict
    return {u'basic': {u'name': name, u'is_dir': True}}


class VersionView(ReadOnlyView):
    """
    A view into a MultiversionBundleFS that exposes only a single version of
    the bundle and its components.
    """

    def __init__(self, bundle_lidvid, versioned_view):
        # type: (LIDVID, MultiversionBundleFS) -> None
        assert bundle_lidvid.lid().is_bundle_lid()
        assert versioned_view.exists(
            lidvid_to_contents_directory_path(bundle_lidvid))
        self._bundle_lidvid = bundle_lidvid
        self._bundle_id = bundle_lidvid.lid().bundle_id
        self._version_id = bundle_lidvid.vid().__str__()
        self._legacy_fs = versioned_view
        self._lid_to_vid_dict = None
        # type: Dict[str, VID]
        ReadOnlyView.__init__(self, versioned_view)

    def _to_legacy_path(self, path):
        # type: (unicode) -> Tuple[str, unicode]
        path = abspath(normpath(path))

        def add_path_segment(legacy_path, new_segment):
            # type: (Tuple[str,unicode], unicode) -> Tuple[str,unicode]
            if legacy_path == ('r', u'/'):
                if new_segment == self._bundle_id:
                    return ('d',
                            join(u'/',
                                 self._bundle_id,
                                 version_id_to_dir_name(self._version_id)))
                else:
                    raise ResourceNotFound(path)
            elif legacy_path[0] == 'f':
                return DirectoryExpected(path)
            else:
                subdir_dict, files = \
                    self._legacy_fs.directory_contents(legacy_path[1])
                if new_segment in files:
                    return ('f', join(legacy_path[1], new_segment))
                try:
                    version_id = subdir_dict[new_segment]
                    new_path = join(dirname(legacy_path[1]),
                                    new_segment,
                                    version_id_to_dir_name(version_id))
                    return ('d', new_path)
                except KeyError:
                    raise ResourceNotFound(path)

        return reduce(add_path_segment, iteratepath(path), ('r', u'/'))

    def getinfo(self, path, namespaces=None):
        type, legacy_path = self._to_legacy_path(path)
        if type == 'd':
            return Info(_make_raw_dir_info(basename(path)))
        elif type == 'f':
            return self._legacy_fs.getinfo(legacy_path, namespaces=namespaces)
        elif type == 'r':
            return Info(_make_raw_dir_info(u'/'))
        assert False, 'uncaught case: %s' % type

    def listdir(self, path):
        type, legacy_path = self._to_legacy_path(path)
        if type == 'd':
            dirs, files = self._legacy_fs.directory_contents(legacy_path)
            return dirs.keys() + files
        elif type == 'f':
            raise DirectoryExpected(path)
        elif type == 'r':
            return [self._bundle_id]
        else:
            assert False, 'uncaught case: %s' % type

    def openbin(self, path, mode="r", buffering=-1, **options):
        type, legacy_path = self._to_legacy_path(path)
        if type == 'f':
            return self._legacy_fs.openbin(
                legacy_path, mode=mode, buffering=buffering, **options)
        elif type in ['d', 'r']:
            raise FileExpected(self.path)
        else:
            assert False, 'uncaught case: %s' % type

    def lid_to_vid(self, lid):
        """
        Returns the VID of a LID that appears in the VersionView.
        """
        # type: (LID) -> VID
        if not self._lid_to_vid_dict:
            # Initialize the dictionary.
            d = dict()
            bundle_lidvid = self._bundle_lidvid
            bundle_lid = bundle_lidvid.lid()
            bundle_vid = bundle_lidvid.vid()
            # TODO Only string keys?
            d[str(bundle_lid)] = bundle_vid
            bundle_subdirs = read_subdir_versions_from_directory(
                self._legacy_fs,
                join(u'/', bundle_lid.bundle_id, vid_to_dir_name(bundle_vid)))
            for coll_id, coll_vid in bundle_subdirs.items():
                collection_lid = '%s:%s' % (bundle_lid, coll_id)
                d[str(collection_lid)] = VID(coll_vid)
                collection_subdirs = read_subdir_versions_from_directory(
                    self._legacy_fs,
                    join(u'/', bundle_lid.bundle_id,
                         coll_id, vid_to_dir_name(VID(coll_vid))))
                for prod_id, prod_vid in collection_subdirs.items():
                    product_lid = '%s:%s' % (collection_lid, prod_id)
                    d[str(product_lid)] = VID(prod_vid)
            self._lid_to_vid_dict = d

        return self._lid_to_vid_dict[str(lid)]

    def directory_to_lidvid(self, dir):
        # type: (unicode) -> LIDVID
        lid = VersionView.directory_to_lid(dir)
        vid = self.lid_to_vid(lid)
        return LIDVID.create_from_lid_and_vid(lid, vid)

    @staticmethod
    def directory_to_lid(dir):
        return LID.create_from_parts(iteratepath(dir))


def layered():
    from fs.osfs import OSFS
    from pdart.fs.InitialVersionedView import InitialVersionedView
    osfs = OSFS('/Users/spaceman/Desktop/Archive/hst_11972')
    ivv = InitialVersionedView('hst_11972', osfs)
    vv = VersionView('urn:nasa:pds:hst_11972::1', ivv)
    return vv
