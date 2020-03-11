from collections import MutableMapping
from typing import TYPE_CHECKING, cast

import fs.path
from fs.subfs import SubFS
from fs.tempfs import TempFS

from pdart.fs.multiversioned.SubdirVersions import (
    SUBDIR_VERSIONS_FILENAME,
    read_subdir_versions_from_directory,
    write_subdir_versions_to_directory,
)
from pdart.fs.multiversioned.VersionContents import VersionContents

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator, List, Optional, Set
    from fs.base import FS

    IS_NEW_TEST = Callable[[LIDVID, VersionContents, Multiversioned], bool]

############################################################


def doc_filter(filepath):
    # type: (unicode) -> bool
    return fs.path.splitext(filepath)[1] in [u".apt", u".pdf", u".pro", u".prop"]


def fits_filter(filepath):
    # type: (unicode) -> bool
    return fs.path.splitext(filepath)[1] == u".fits"


def std_is_new(lidvid, contents, mv):
    # type: (LIDVID, VersionContents, Multiversioned) -> None
    lid = lidvid.lid()
    if lid.is_collection_lid() and lid.collection_id == "document":
        filt = doc_filter
    else:
        filt = fits_filter
    return mv[lidvid].filter_filepaths(filt) != contents


############################################################


def lid_path(lid):
    # type: (LID) -> unicode
    """
    Return the directory path co`rresponding to this LID.
    """

    def lid_to_segments(lid):
        # type: (LID) -> List[unicode]
        """
        Extract the segments (bundle, collection, product) from the
        LID and return as a list.
        """
        res = [lid.bundle_id]
        if lid.collection_id:
            res.append(lid.collection_id)
        if lid.product_id:
            res.append(lid.product_id)
        return [unicode(id) for id in res]

    dir_segments = lid_to_segments(lid)
    return fs.path.abspath(apply(fs.path.join, dir_segments))


def lidvid_path(lidvid):
    # type: (LIDVID) -> unicode
    """
    Return the directory path corresponding to this LIDVID.
    """

    def vid_to_dir_part(vid):
        # type: (VID) -> unicode
        """
        Convert a VID to a directory name.
        """
        return "v$%s" % str(vid)

    lid = lidvid.lid()
    dir = lid_path(lid)
    vid_bit = vid_to_dir_part(lidvid.vid())
    return fs.path.join(dir, vid_bit)


############################################################


def is_sub_lid(parent_lid, child_lid):
    # type: (LID, LID) -> bool
    """
    Return true iff the first LID is the parent of the second.
    """
    if child_lid.is_bundle_lid():
        return False
    else:
        return child_lid.parent_lid() == parent_lid


############################################################


class Multiversioned(MutableMapping):
    """
    Provides functionality to store multiple bundles into a single
    pyfilesystem.  Represents the wrapped filesystem as a mapping from
    LIDVIDs to VersionContents.
    """

    def __init__(self, fs):
        # type: (FS) -> None
        self.fs = fs

    def make_lidvid_dir(self, lidvid):
        # type: (LIDVID) -> unicode
        dir_path = lidvid_path(lidvid)
        self.fs.makedirs(dir_path, None, True)
        return dir_path

    def lidvids(self):
        # type: () -> Set[LIDVID]
        return set(self.__iter__())

    def latest_lidvid(self, lid):
        # type: (LID) -> Optional[LIDVID]
        matching = [lidvid for lidvid in self.lidvids() if lidvid.lid() == lid]
        if matching:
            return max(matching)
        else:
            return None

    def next_major_lidvid(self, lid):
        # type: (LID) -> LIDVID
        latest = self.latest_lidvid(lid)
        if latest:
            return latest.next_major_lidvid()
        else:
            return LIDVID.create_from_lid_and_vid(lid, VID("1.0"))

    def next_minor_lidvid(self, lid):
        # type: (LID) -> LIDVID
        latest = self.latest_lidvid(lid)
        if latest:
            return latest.next_minor_lidvid()
        else:
            return LIDVID.create_from_lid_and_vid(lid, VID("1.0"))

    def create_version_view(self, lid):
        # type: (LID) -> FS
        lidvid = self.latest_lidvid(lid)
        if lidvid is None:
            # It's only read, not written to, and the Multiversioned
            # is empty (at least for this bundle). We can return
            # anything that's empty, so:
            return TempFS()
        else:
            from multiversioned.VersionView import VersionView

            return VersionView(self, lidvid)

    def add_contents_if(self, is_new, lid, contents, minor_change=False):
        # type: (IS_NEW_TEST, LID, VersionContents, bool) -> LIDVID
        lidvid = self.latest_lidvid(lid)

        if lidvid is None or is_new(lidvid, contents, self):
            if minor_change:
                new_lidvid = self.next_minor_lidvid(lid)
            else:
                new_lidvid = self.next_major_lidvid(lid)
            self[new_lidvid] = contents
            return new_lidvid
        else:
            return lidvid

    ############################################################

    def update_from_single_version(self, is_new, single_version_fs):
        # type: (IS_NEW_TEST, FS) -> bool

        # TODO This import is circular; that's why I have it here
        # inside the function.  But there must be a better way to
        # structure.
        from multiversioned.VersionView import is_segment, strip_segment, vv_lid_path

        # TODO Note that this makes assumptions about the source
        # filesystem format.  Document them.

        def update_from_lid(lid):
            # type: (LID) -> LIDVID
            path = vv_lid_path(lid)
            child_lidvids = {
                update_from_lid(lid.extend_lid(strip_segment(name)))
                for name in single_version_fs.listdir(path)
                if is_segment(name)
            }
            sfs = SubFS(single_version_fs, path)
            filepaths = {
                filepath for filepath in sfs.walk.files() if "$" not in filepath
            }
            contents = VersionContents(True, child_lidvids, sfs, filepaths)
            return self.add_contents_if(is_new, lid, contents, False)

        bundle_segs = [
            strip_segment(name)
            for name in single_version_fs.listdir(u"/")
            if is_segment(name)
        ]

        # TODO I can't see any reason why there wouldn't be exactly a
        # single segment, but I'm throwing in an assert to let me know
        # if I'm wrong.
        assert len(bundle_segs) == 1

        changed = False

        for bundle_seg in bundle_segs:
            lid = LID.create_from_parts([str(bundle_seg)])
            orig_lidvid = self.latest_lidvid(lid)
            new_lidvid = update_from_lid(lid)
            changed = changed or orig_lidvid != new_lidvid

        return changed

    ############################################################

    def __contains__(self, lidvid):
        # type: (Any) -> bool
        if isinstance(lidvid, LIDVID):
            return self.fs.isdir(lidvid_path(lidvid))
        else:
            return False

    def __delitem__(self, lidvid):
        # type: (LIDVID) -> None
        raise NotImplementedError("deletion is not allowed for Multiversioneds")

    def __getitem__(self, lidvid):
        # type: (LIDVID) -> VersionContents
        dirpath = lidvid_path(lidvid)
        if not self.fs.isdir(dirpath):
            raise KeyError(lidvid)
        d = read_subdir_versions_from_directory(self.fs, dirpath)

        def make_sub_lidvid(seg, vid_part):
            # type: (str, str) -> LIDVID
            lid_parts = lidvid.lid().parts()
            lid_parts.append(seg)
            return LIDVID.create_from_lid_and_vid(
                LID.create_from_parts(lid_parts), VID(vid_part)
            )

        lidvids = {make_sub_lidvid(segment, vid) for segment, vid in d.items()}
        sub_fs = SubFS(self.fs, dirpath)
        filepaths = set(sub_fs.walk.files(exclude=[SUBDIR_VERSIONS_FILENAME]))
        return VersionContents(True, lidvids, sub_fs, filepaths)

    def __iter__(self):
        # type: () -> Iterator[LIDVID]
        for dir in self.fs.walk.dirs():
            parts = fs.path.parts(dir)
            if parts[-1].startswith("v$"):
                vid_part = str(parts[-1][2:])
                lid_parts = [str(p) for p in parts[1:-1]]
                yield LIDVID.create_from_lid_and_vid(
                    LID.create_from_parts(lid_parts), VID(vid_part)
                )

    def __len__(self):
        res = 0
        for dir in self.fs.walk.dirs():
            parts = fs.path.parts(dir)
            if parts[-1].startswith("v$"):
                res += 1
        return res

    def __setitem__(self, lidvid, contents):
        # type: (LIDVID, VersionContents) -> None
        if lidvid in self:
            # illegal to set twice
            raise IndexError("Repeated setting at %s" % lidvid)
        # check that contents aren't contradictory
        lidvid_parts = lidvid.lid().parts()

        d = {}
        for sub_lidvid in contents.lidvids():
            sub_parts = sub_lidvid.lid().parts()
            d[sub_parts[-1]] = str(sub_lidvid.vid())
            if sub_parts[:-1] != lidvid_parts:
                raise ValueError(
                    "LIDVID %s in contents cannot be "
                    "a child of index LIDVID %s" % (sub_lidvid, lidvid)
                )
        lidvid_dir = self.make_lidvid_dir(lidvid)
        if d:
            write_subdir_versions_to_directory(self.fs, lidvid_dir, d)

        for src_filepath in contents.filepaths:
            dst_filepath = fs.path.join(lidvid_dir, fs.path.relpath(src_filepath))
            self.fs.makedirs(fs.path.dirname(dst_filepath), None, True)
            fs.copy.copy_file(contents.fs, src_filepath, self.fs, dst_filepath)
        assert self.fs.isdir(lidvid_path(lidvid))
        assert lidvid in self
