"""
A view into a MultiversionBundleFS that exposes only a single version
of the bundle and its components.
"""
from fs.errors import DirectoryExpected, FileExpected, ResourceNotFound
from fs.info import Info
from fs.mode import Mode
from fs.path import abspath, basename, iteratepath, normpath, split

from pdart.fs.DirUtils import lidvid_to_dir
from pdart.fs.ISingleVersionBundleFS import ISingleVersionBundleFS
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.ReadOnlyView import ReadOnlyView
from pdart.fs.SubdirVersions import *
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from typing import Dict, Tuple
    from fs.osfs import OSFS


def _make_raw_dir_info(name):
    # type: (unicode) -> Dict
    return {u'basic': {u'name': name, u'is_dir': True}}


class VersionView(ReadOnlyView, ISingleVersionBundleFS):
    """
    A view into a MultiversionBundleFS that exposes only a single version of
    the bundle and its components.
    """

    def __init__(self, bundle_lidvid, versioned_view):
        # type: (LIDVID, MultiversionBundleFS) -> None
        assert bundle_lidvid.lid().is_bundle_lid()
        assert versioned_view.exists(lidvid_to_dir(bundle_lidvid))
        self._bundle_lidvid = bundle_lidvid
        self._bundle_id = bundle_lidvid.lid().bundle_id
        self._version_id = bundle_lidvid.vid().__str__()
        self._legacy_fs = versioned_view
        ReadOnlyView.__init__(self, versioned_view)

    def _to_legacy_path(self, path, writing):
        # type: (unicode, bool) -> Tuple[str, unicode]
        path = abspath(normpath(path))

        def add_path_segment(legacy_path, new_segment):
            # type: (Tuple[str,unicode], unicode) -> Tuple[str,unicode]
            if legacy_path == ('r', u'/'):
                if new_segment == self._bundle_id:
                    return 'd', lidvid_to_dir(self._bundle_lidvid)
                else:
                    raise ResourceNotFound(path)
            elif legacy_path[0] == 'f':
                raise DirectoryExpected(path)
            elif legacy_path[0] == 'd':
                # 'd' path: legacy path is a directory
                subdir_dict, files = \
                    self._legacy_fs.directory_contents(legacy_path[1])
                if new_segment in files:
                    return 'f', join(legacy_path[1], new_segment)
                try:
                    version_id = subdir_dict[str(new_segment)]
                except KeyError:
                    if writing:
                        version_id = '1'
                    else:
                        raise ResourceNotFound(path)
                parts = iteratepath(legacy_path[1])
                parts[-1] = new_segment
                lid = LID.create_from_parts(parts)
                lidvid = LIDVID.create_from_lid_and_vid(lid, VID(version_id))
                return 'd', lidvid_to_dir(lidvid)
            else:
                raise Exception('unexpected branch: legacy_path == %s' %
                                str(legacy_path))

        def build_path(path_to_build):
            # type: (unicode) -> Tuple[str, unicode]
            return reduce(add_path_segment, iteratepath(path_to_build),
                          ('r', u'/'))

        if writing:
            # Don't require the last segment to exist.  Just calculate
            # its parent directory and add the possibly non-existent filename
            # to it.
            (parent, base) = split(path)
            (file_type, legacy_parent) = build_path(parent)
            if file_type == 'd':
                # TODO Should check I'm not overwriting a directory.
                return 'f', join(legacy_parent, base)
            else:
                raise DirectoryExpected(parent)
        else:
            return build_path(path)

    def getinfo(self, path, namespaces=None):
        file_type, legacy_path = self._to_legacy_path(path, False)
        if file_type == 'd':
            return Info(_make_raw_dir_info(basename(path)))
        elif file_type == 'f':
            return self._legacy_fs.getinfo(legacy_path, namespaces=namespaces)
        elif file_type == 'r':
            return Info(_make_raw_dir_info(u'/'))
        assert False, 'uncaught case: %s' % file_type

    def listdir(self, path):
        file_type, legacy_path = self._to_legacy_path(path, False)
        if file_type == 'd':
            dirs, files = self._legacy_fs.directory_contents(legacy_path)
            return dirs.keys() + files
        elif file_type == 'f':
            raise DirectoryExpected(path)
        elif file_type == 'r':
            return [self._bundle_id]
        else:
            assert False, 'uncaught case: %s' % file_type

    def openbin(self, path, mode="r", buffering=-1, **options):
        writing = Mode(mode).writing
        file_type, legacy_path = self._to_legacy_path(path, writing)
        if file_type == 'f':
            return self._legacy_fs.openbin(
                legacy_path, mode=mode, buffering=buffering, **options)
        elif file_type in ['d', 'r']:
            raise FileExpected(path)
        else:
            assert False, 'uncaught case: %s' % file_type

    def bundle_lidvid(self):
        # type: () -> LIDVID
        return self._bundle_lidvid

    def lid_to_vid(self, lid):
        # type: (LID) -> VID
        """
        Returns the VID of a LID that appears in the VersionView.
        """
        if lid.is_bundle_lid():
            assert lid == self.bundle_lidvid().lid(), \
                "%s != %s" % (lid, self.bundle_lidvid().lid())
            return self.bundle_lidvid().vid()
        elif lid.is_collection_lid():
            bundle_lid = lid.parent_lid()
            bundle_vid = self.lid_to_vid(bundle_lid)
            bundle_lidvid = LIDVID.create_from_lid_and_vid(bundle_lid,
                                                           bundle_vid)
            bundle_subdirs = read_subdir_versions_from_directory(
                self._legacy_fs, lidvid_to_dir(bundle_lidvid)
            )
            return VID(bundle_subdirs[lid.collection_id])
        elif lid.is_product_lid():
            collection_lid = lid.parent_lid()
            collection_vid = self.lid_to_vid(collection_lid)
            collection_lidvid = LIDVID.create_from_lid_and_vid(
                collection_lid,
                collection_vid)
            collection_subdirs = read_subdir_versions_from_directory(
                self._legacy_fs, lidvid_to_dir(collection_lidvid))
            return VID(collection_subdirs[lid.product_id])
        else:
            assert False, 'impossible case: %r' % lid


def _layered():
    from fs.osfs import OSFS
    from pdart.fs.InitialVersionedView import InitialVersionedView
    osfs = OSFS('/Users/spaceman/Desktop/Archive/hst_11972')
    ivv = InitialVersionedView('hst_11972', osfs)
    vv = VersionView(LIDVID('urn:nasa:pds:hst_11972::1'), ivv)
    return vv
