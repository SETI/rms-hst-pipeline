from functools import reduce
from typing import Any, BinaryIO, Collection, List, Mapping, Optional

import fs.mode
import fs.path
from fs.base import FS
from fs.errors import ResourceNotFound, ResourceReadOnly
from fs.info import Info
from fs.permissions import Permissions
from fs.subfs import SubFS

from pdart.fs.multiversioned.SubdirVersions import (
    SUBDIR_VERSIONS_FILENAME,
    read_subdir_versions_from_directory,
)
from pdart.fs.multiversioned.VersionContents import VersionContents
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

_INFO = Mapping[str, Mapping[str, Any]]


def is_segment(seg: str) -> bool:
    return seg[-1] == "$"


def strip_segment(seg: str) -> str:
    return str(seg[:-1])


def vv_lid_path(lid: LID) -> str:
    parts = lid.parts()
    return fs.path.join(*[part + "$" for part in parts])


class VersionView(FS):
    def __init__(self, mv: Any, lidvid: LIDVID) -> None:
        # TODO mv should be Multiversioned, but there are circular
        # imports.  Fix this.
        from pdart.fs.multiversioned.Multiversioned import Multiversioned

        assert isinstance(mv, Multiversioned)
        FS.__init__(self)
        assert lidvid in mv
        self.multiversioned = mv
        self.lidvid = lidvid

    def lid_to_lidvid(self, lid: LID) -> LIDVID:
        vv_path = vv_lid_path(lid)
        mv_path = self.transform_vv_path(vv_path)
        parts = [str(p) for p in fs.path.parts(mv_path)]
        lid = LID.create_from_parts(parts[1:-1])
        vid_part = parts[-1]
        assert vid_part.startswith("v$")
        vid = VID(vid_part[2:])
        return LIDVID.create_from_lid_and_vid(lid, vid)

    ############################################################

    def transform_vv_path(self, vv_path: str) -> str:
        # Since this is a read-only view, we never create anything, so
        # we have no use for nonexistent filepaths and can raise an
        # exception whenever they're sought.  Right?  TODO Check this.
        vv_parts = fs.path.parts(fs.path.abspath(vv_path))[1:]

        first = [True]

        def add_vv_part_to_mv_path(mv_path: str, vv_part: str) -> str:
            if first[0]:
                first[0] = False
                if is_segment(vv_part):
                    if vv_part[:-1] == self.lidvid.lid().bundle_id:
                        # TODO this is a hack; fix it.
                        from pdart.fs.multiversioned.Multiversioned import lidvid_path

                        return lidvid_path(self.lidvid)
                    else:
                        raise ResourceNotFound(vv_path)
                else:
                    raise ResourceNotFound(vv_path)
            else:
                if is_segment(vv_part):
                    d = read_subdir_versions_from_directory(
                        self.multiversioned.fs, mv_path
                    )
                    next_vid = d[str(vv_part[:-1])]
                    return fs.path.join(
                        fs.path.dirname(mv_path), vv_part[:-1], "v$" + next_vid
                    )

                else:
                    return fs.path.join(mv_path, vv_part)

        return reduce(add_vv_part_to_mv_path, vv_parts, "/")

    ############################################################

    def getinfo(self, path: str, namespaces: Optional[Collection[str]] = None) -> Info:
        self.check()
        mv_path = self.transform_vv_path(path)
        name = fs.path.basename(path)
        is_dir = self.multiversioned.fs.isdir(mv_path)
        d = {"basic": {"name": name, "is_dir": is_dir}}
        return Info(d)

    def listdir(self, path: str) -> List[str]:
        self.check()

        if path == "/":
            return [self.lidvid.lid().bundle_id + "$"]

        mv_path = self.transform_vv_path(path)
        vv_res = self.multiversioned.fs.listdir(mv_path)
        if SUBDIR_VERSIONS_FILENAME in vv_res:
            vv_res.remove(SUBDIR_VERSIONS_FILENAME)
        d = read_subdir_versions_from_directory(self.multiversioned.fs, mv_path)
        vv_res.extend([k + "$" for k in d])

        return sorted(vv_res)

    def openbin(
        self, path: str, mode: str = "r", buffering: int = -1, **options: Any
    ) -> BinaryIO:
        self.check()
        if fs.mode.Mode(mode).writing:
            raise ResourceReadOnly(path)

        mv_path = self.transform_vv_path(path)
        return self.multiversioned.fs.openbin(mv_path, mode, buffering, **options)

    ############################################################

    def makedir(
        self,
        path: str,
        permissions: Optional[Permissions] = None,
        recreate: bool = False,
    ) -> SubFS[FS]:
        self.check()
        raise ResourceReadOnly(path)

    def remove(self, path: str) -> None:
        self.check()
        raise ResourceReadOnly(path)

    def removedir(self, path: str) -> None:
        self.check()
        raise ResourceReadOnly(path)

    def setinfo(self, path: str, info: _INFO) -> None:
        self.check()
        raise ResourceReadOnly(path)

    def getsyspath(self, path: str) -> str:
        self.check()
        if self.isfile(path):
            mv_path = self.transform_vv_path(path)
            return self.multiversioned.fs.getsyspath(mv_path)
        else:
            raise fs.errors.NoSysPath(path=path)

    ############################################################

    def __getitem__(self, lid: LID) -> VersionContents:
        dirpath = vv_lid_path(lid)
        if not self.isdir(dirpath):
            raise KeyError(lid)
        subcomps = {
            lid.extend_lid(str(name)[:-1])
            for name in self.listdir(dirpath)
            if is_segment(name)
        }
        return VersionContents.create_from_lids_and_dirpath(subcomps, self, dirpath)
