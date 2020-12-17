from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, List, Optional, Set, cast

import fs.path
from fs.base import FS
from fs.subfs import SubFS
from fs.tempfs import TempFS

from pdart.documents.Downloads import DOCUMENT_SUFFIXES
from pdart.fs.multiversioned.SubdirVersions import (
    SUBDIR_VERSIONS_FILENAME,
    read_subdir_versions_from_directory,
    write_subdir_versions_to_directory,
)
from pdart.fs.multiversioned.VersionContents import VersionContents
from pdart.fs.multiversioned.VersionView import VersionView
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

# An IS_NEW_TEST is a function on a proposed LIDVID with the given
# VersionContents, and a Multiversioned that returns True iff the
# contents would be changed from the last version of the LID.
IS_NEW_TEST = Callable[[LIDVID, VersionContents, "Multiversioned"], bool]

############################################################


def doc_filter(filepath: str) -> bool:
    return fs.path.splitext(filepath)[1] in DOCUMENT_SUFFIXES


def fits_filter(filepath: str) -> bool:
    return fs.path.splitext(filepath)[1] == ".fits"


def std_is_new(lidvid: LIDVID, contents: VersionContents, mv: "Multiversioned") -> bool:
    """
    The standard IS_NEW_TEST function.  If it's a document collection,
    it checks the document files for changes; otherwise, it checks
    only FITS files.
    """
    lid = lidvid.lid()
    if lid.is_collection_lid() and lid.collection_id == "document":
        filt = doc_filter
    else:
        filt = fits_filter
    return mv[lidvid].filter_filepaths(filt) != contents


############################################################


def lid_path(lid: LID) -> str:
    """
    Return the directory path corresponding to this LID.
    """

    def lid_to_segments(lid: LID) -> List[str]:
        """
        Extract the segments (bundle, collection, product) from the
        LID and return as a list.
        """
        res = [lid.bundle_id]
        if lid.collection_id:
            res.append(lid.collection_id)
        if lid.product_id:
            res.append(lid.product_id)
        return [str(id) for id in res]

    dir_segments = lid_to_segments(lid)
    return fs.path.abspath(fs.path.join(*dir_segments))


def lidvid_path(lidvid: LIDVID) -> str:
    """
    Return the directory path corresponding to this LIDVID.
    """

    def vid_to_dir_part(vid: VID) -> str:
        """
        Convert a VID to a directory name.
        """
        return f"v${vid}"

    lid = lidvid.lid()
    dir = lid_path(lid)
    vid_bit = vid_to_dir_part(lidvid.vid())
    return fs.path.join(dir, vid_bit)


############################################################


def is_sub_lid(parent_lid: LID, child_lid: LID) -> bool:
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

    def __init__(self, fs: FS) -> None:
        self.fs = fs

    def __str__(self) -> str:
        return f"Multiversioned({self.fs})"

    def make_lidvid_dir(self, lidvid: LIDVID) -> str:
        dir_path = lidvid_path(lidvid)
        self.fs.makedirs(dir_path, None, True)
        return dir_path

    def lidvids(self) -> Set[LIDVID]:
        return set(self.__iter__())

    def latest_lidvid(self, lid: LID) -> Optional[LIDVID]:
        matching = [lidvid for lidvid in self.lidvids() if lidvid.lid() == lid]
        if matching:
            return max(matching)
        else:
            return None

    def next_major_lidvid(self, lid: LID) -> LIDVID:
        latest = self.latest_lidvid(lid)
        if latest:
            return latest.next_major_lidvid()
        else:
            return LIDVID.create_from_lid_and_vid(lid, VID("1.0"))

    def next_minor_lidvid(self, lid: LID) -> LIDVID:
        latest = self.latest_lidvid(lid)
        if latest:
            return latest.next_minor_lidvid()
        else:
            return LIDVID.create_from_lid_and_vid(lid, VID("1.0"))

    def create_version_view(self, lid: LID) -> VersionView:
        lidvid = self.latest_lidvid(lid)
        if lidvid is None:
            # It's only read, not written to, and the Multiversioned
            # is empty (at least for this bundle). We can return
            # anything that's empty, so:
            return cast(VersionView, TempFS())
        else:
            return VersionView(self, lidvid)

    def add_contents_if(
        self,
        is_new: IS_NEW_TEST,
        lid: LID,
        contents: VersionContents,
        minor_change: bool = False,
    ) -> LIDVID:
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

    def update_from_single_version(
        self, is_new: IS_NEW_TEST, single_version_fs: FS
    ) -> bool:
        # TODO This import is circular; that's why I have it here
        # inside the function.  But there must be a better way to
        # structure.
        from pdart.fs.multiversioned.VersionView import (
            is_segment,
            strip_segment,
            vv_lid_path,
        )

        # TODO Note that this makes assumptions about the source
        # filesystem format.  Document them.

        def update_from_lid(lid: LID) -> LIDVID:
            # Find the path corresponding to this LID.
            path = vv_lid_path(lid)

            # First, update all the children recursively.  Get their
            # LIDs by extending this LID with the names of the
            # subdirectories of path.  That handles directories.
            child_lidvids: Set[LIDVID] = {
                update_from_lid(lid.extend_lid(strip_segment(name)))
                for name in single_version_fs.listdir(path)
                if is_segment(name)
            }

            # Now look at files.  We create a VersionContents object
            # from the set of new LIDVIDs and all the files contained
            # in the component's directory.
            contents = VersionContents.create_from_lidvids_and_dirpath(
                child_lidvids, single_version_fs, path
            )

            # Now we ask the Multiversioned to insert these contents
            # as a new version if needed.  It returns the new LIDVID
            # if a new LIDVID is needed, otherwise it returns the old
            # one.
            return self.add_contents_if(is_new, lid, contents, False)

        bundle_segs = [
            strip_segment(name)
            for name in single_version_fs.listdir("/")
            if is_segment(name)
        ]

        # TODO I can't see any reason why there wouldn't be exactly a
        # single segment, but I'm throwing in an assert to let me know
        # if I'm wrong.
        assert len(bundle_segs) == 1

        changed = False

        for bundle_seg in bundle_segs:
            lid = LID.create_from_parts([str(bundle_seg)])
            orig_lidvid: Optional[LIDVID] = self.latest_lidvid(lid)
            new_lidvid: LIDVID = update_from_lid(lid)
            changed = changed or new_lidvid != orig_lidvid

        return changed

    ############################################################

    def __contains__(self, lidvid: Any) -> bool:
        if isinstance(lidvid, LIDVID):
            return self.fs.isdir(lidvid_path(lidvid))
        else:
            return False

    def __delitem__(self, lidvid: LIDVID) -> None:
        raise NotImplementedError("deletion is not allowed for Multiversioneds")

    def __getitem__(self, lidvid: LIDVID) -> VersionContents:
        dirpath = lidvid_path(lidvid)
        if not self.fs.isdir(dirpath):
            raise KeyError(lidvid)
        d = read_subdir_versions_from_directory(self.fs, dirpath)

        def make_sub_lidvid(seg: str, vid_part: str) -> LIDVID:
            lid_parts = lidvid.lid().parts()
            lid_parts.append(seg)
            return LIDVID.create_from_lid_and_vid(
                LID.create_from_parts(lid_parts), VID(vid_part)
            )

        lidvids = {make_sub_lidvid(segment, vid) for segment, vid in list(d.items())}
        sub_fs = SubFS(self.fs, dirpath)
        filepaths = set(sub_fs.walk.files(exclude=[SUBDIR_VERSIONS_FILENAME]))
        return VersionContents.create_from_lidvids(lidvids, sub_fs, filepaths)

    def __iter__(self) -> Iterator[LIDVID]:
        for dir in self.fs.walk.dirs():
            parts = fs.path.parts(dir)
            if parts[-1].startswith("v$"):
                vid_part = str(parts[-1][2:])
                lid_parts = [str(p) for p in parts[1:-1]]
                yield LIDVID.create_from_lid_and_vid(
                    LID.create_from_parts(lid_parts), VID(vid_part)
                )

    def __len__(self) -> int:
        res = 0
        for dir in self.fs.walk.dirs():
            parts = fs.path.parts(dir)
            if parts[-1].startswith("v$"):
                res += 1
        return res

    def __setitem__(self, lidvid: LIDVID, contents: VersionContents) -> None:
        if lidvid in self:
            # illegal to set twice
            raise IndexError(f"Repeated setting at {lidvid}")
        # check that contents aren't contradictory
        lidvid_parts = lidvid.lid().parts()

        d = {}
        for sub_lidvid in contents.lidvids():
            sub_parts = sub_lidvid.lid().parts()
            d[sub_parts[-1]] = str(sub_lidvid.vid())
            if sub_parts[:-1] != lidvid_parts:
                raise ValueError(
                    f"LIDVID {sub_lidvid} in contents cannot be "
                    f"a child of index LIDVID {lidvid}"
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
