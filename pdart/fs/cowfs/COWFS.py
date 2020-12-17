from typing import (
    Any,
    BinaryIO,
    Collection,
    Generator,
    List,
    Mapping,
    Optional,
)

import fs.copy
import fs.errors
import fs.path
import fs.wrap
from fs.base import FS
from fs.info import Info
from fs.mode import Mode
from fs.permissions import Permissions
from fs.subfs import SubFS
from fs.tempfs import TempFS

_INFO_DICT = Mapping[str, Mapping[str, object]]

NO_LAYER = 0  # exists in deletion layer
BASE_LAYER = 1  # exists in base (original, read-only) layer
ADD_LAYER = 2  # exists in addition layer
ROOT_LAYER = 3  # not really a layer but a special case

layer_names = ["NO_LAYER", "BASE_LAYER", "ADD_LAYER", "ROOT_LAYER"]

DEL = "__DELETED__"


def _deletions_invariant(filesys: FS) -> None:
    """
    The only files the given filesystem contains all are named DEL.
    """
    names = {fs.path.basename(file) for file in filesys.walk.files()}
    assert not names or names == {DEL}


def paths(filesys: FS) -> Generator[str, None, None]:
    """
    All the directory and file names in the filesystem.
    """
    for dir in filesys.walk.dirs():
        yield dir
    for file in filesys.walk.files():
        yield file


def del_path(path: str) -> str:
    """
    Create the filepath for the deletion marker file.
    """
    return fs.path.combine(path, DEL)


class COWFS(FS):
    def __init__(
        self,
        base_fs: FS,
        additions_fs: Optional[FS] = None,
        deletions_fs: Optional[FS] = None,
    ) -> None:
        FS.__init__(self)
        if additions_fs:
            self.additions_fs = additions_fs
        else:
            self.additions_fs = TempFS()

        if deletions_fs:
            _deletions_invariant(deletions_fs)
            self.deletions_fs = deletions_fs
        else:
            self.deletions_fs = TempFS()

        self.original_base_fs = base_fs
        self.base_fs = fs.wrap.read_only(base_fs)

        self.invariant()

    @staticmethod
    def create_cowfs(
        base_fs: FS, read_write_layer: FS, recreate: bool = False
    ) -> "COWFS":
        additions_fs = read_write_layer.makedir("/additions", recreate=recreate)
        deletions_fs = read_write_layer.makedir("/deletions", recreate=recreate)

        return COWFS(base_fs, additions_fs, deletions_fs)

    def __str__(self) -> str:
        return (
            f"COWFS({self.original_base_fs}, "
            f"{self.additions_fs}, "
            f"{self.deletions_fs})"
        )

    def __repr__(self) -> str:
        return (
            f"COWFS({self.original_base_fs!r}, "
            f"{self.additions_fs!r}, "
            f"{self.deletions_fs!r})"
        )

    ############################################################

    def invariant(self) -> bool:
        assert self.additions_fs
        assert self.deletions_fs
        assert self.base_fs

        _deletions_invariant(self.deletions_fs)

        additions_paths = set(paths(self.additions_fs))
        deletions_paths = {
            fs.path.dirname(file) for file in self.deletions_fs.walk.files()
        }
        assert (
            additions_paths <= deletions_paths
        ), f"""Additions_paths {additions_paths}
is not a subset of
deletions_path {deletions_paths}.
Extras are {additions_paths - deletions_paths}.
"""

        return True

    def is_deletion(self, path: str) -> bool:
        """
        Is the path marked in the deletions_fs"
        """
        return self.deletions_fs.exists(del_path(path))

    def mark_deletion(self, path: str) -> None:
        """
        Mark the path in the deletions_fs.
        """
        self.deletions_fs.makedirs(path, None, True)
        self.deletions_fs.touch(del_path(path))

    def makedirs_mark_deletion(
        self,
        path: str,
        permissions: Optional[Permissions] = None,
        recreate: bool = False,
    ) -> None:
        for p in fs.path.recursepath(path)[:-1]:
            self.additions_fs.makedirs(p, permissions=permissions, recreate=True)
            self.mark_deletion(p)
        self.additions_fs.makedir(path, permissions=permissions, recreate=recreate)
        self.mark_deletion(path)

    def layer(self, path: str) -> int:
        """
        Get the layer on which the file lives, or ROOT_LAYER if it's the
        root path.
        """

        if path == "/":
            return ROOT_LAYER
        if self.additions_fs.exists(path):
            return ADD_LAYER
        elif self.is_deletion(path):
            return NO_LAYER
        elif self.base_fs.exists(path):
            return BASE_LAYER
        else:
            return NO_LAYER

    def copy_up(self, path: str) -> None:
        """
        Copy the file from the base_fs to additions_fs.
        """
        self.makedirs_mark_deletion(fs.path.dirname(path))
        self.mark_deletion(path)
        fs.copy.copy_file(self.base_fs, path, self.additions_fs, path)

    def triple_tree(self) -> None:
        print("base_fs ------------------------------")
        self.base_fs.tree()
        print("additions_fs ------------------------------")
        self.additions_fs.tree()
        print("deletions_fs ------------------------------")
        self.deletions_fs.tree()

    ############################################################
    def getmeta(self, namespace: str = "standard") -> Mapping[str, object]:
        return self.base_fs.getmeta(namespace)

    def getinfo(self, path: str, namespaces: Optional[Collection[str]] = None) -> Info:
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            return self.base_fs.getinfo(path, namespaces)
        elif layer == ADD_LAYER:
            return self.additions_fs.getinfo(path, namespaces)
        elif layer == ROOT_LAYER:
            # TODO implement this
            raw_info = {}
            if namespaces is None or "basic" in namespaces:
                raw_info["basic"] = {"name": "", "is_dir": True}
            return Info(raw_info)
        else:
            assert False, f"unknown layer {layer}"

    def getsyspath(self, path: str) -> str:
        self.check()
        # self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.NoSysPath(path=path)
        elif layer == BASE_LAYER:
            return self.base_fs.getsyspath(path)
        elif layer == ADD_LAYER:
            return self.additions_fs.getsyspath(path)
        elif layer == ROOT_LAYER:
            raise fs.errors.NoSysPath(path=path)
        else:
            assert False, f"unknown layer {layer}"

    def listdir(self, path: str) -> List[str]:
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            return [
                name
                for name in self.base_fs.listdir(path)
                if self.layer(fs.path.join(path, name)) != NO_LAYER
            ]
        elif layer == ADD_LAYER:
            # Get the listing on the additions layer
            names = set(self.additions_fs.listdir(path))
            # Add in the listing on the base layer (if it exists)
            if self.base_fs.isdir(path):
                names |= set(self.base_fs.listdir(path))
            # Return the entries that actually exist
            return [
                name
                for name in list(names)
                if self.layer(fs.path.join(path, name)) != NO_LAYER
            ]
        elif layer == ROOT_LAYER:
            # Get the listing of the root on the additions layer and
            # the base layer.
            names = set(self.additions_fs.listdir("/"))
            names |= set(self.base_fs.listdir("/"))
            # Return the entries that actually exist.
            return [name for name in list(names) if self.layer(name) != NO_LAYER]
        else:
            assert False, f"unknown layer {layer}"

    def makedir(
        self,
        path: str,
        permissions: Optional[Permissions] = None,
        recreate: bool = False,
    ) -> SubFS["COWFS"]:
        self.check()
        self.validatepath(path)

        # Check if it *can* be created.

        # get a normalized parent_dir path.
        parent_dir = fs.path.dirname(fs.path.forcedir(path)[:-1])
        if not parent_dir:
            parent_dir = "/"

        if not self.isdir(parent_dir):
            raise fs.errors.ResourceNotFound(path)

        layer = self.layer(path)
        if layer == NO_LAYER:
            self.makedirs_mark_deletion(
                path, permissions=permissions, recreate=recreate
            )
            return SubFS(self, path)
        elif layer in [BASE_LAYER, ADD_LAYER, ROOT_LAYER]:
            if recreate:
                return SubFS(self, path)
            else:
                # I think this is wrong.  What if it's a file?
                raise fs.errors.DirectoryExists(path)
        else:
            assert False, f"unknown layer {layer}"

    def openbin(
        self, path: str, mode: str = "r", buffering: int = -1, **options: Any
    ) -> BinaryIO:
        self.check()
        self.validatepath(path)

        parent_dir = fs.path.dirname(fs.path.forcedir(path)[:-1])
        if not parent_dir:
            parent_dir = "/"

        if not self.isdir(parent_dir):
            raise fs.errors.ResourceNotFound(path)

        mode_obj = Mode(mode)
        layer = self.layer(path)
        if layer == NO_LAYER:
            if mode_obj.create:
                for p in fs.path.recursepath(path)[:-1]:
                    self.additions_fs.makedirs(p, recreate=True)
                    self.mark_deletion(p)
                self.mark_deletion(path)
                return self.additions_fs.openbin(path, mode, buffering, **options)
            else:
                raise fs.errors.ResourceNotFound(path)
        elif layer == ADD_LAYER:
            self.mark_deletion(path)
            return self.additions_fs.openbin(path, mode, buffering, **options)
        elif layer == BASE_LAYER:
            if mode_obj.writing:
                self.copy_up(path)
                return self.additions_fs.openbin(path, mode, buffering, **options)
            else:
                return self.base_fs.openbin(path, mode, buffering, **options)
        elif layer == ROOT_LAYER:
            raise fs.errors.FileExpected(path)
        else:
            assert False, f"unknown layer {layer}"

    def remove(self, path: str) -> None:
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            if self.base_fs.isfile(path):
                self.mark_deletion(path)
            else:
                raise fs.errors.FileExpected(path)
        elif layer == ADD_LAYER:
            self.additions_fs.remove(path)
            self.mark_deletion(path)
        elif layer == ROOT_LAYER:
            raise fs.errors.FileExpected(path)
        else:
            assert False, f"unknown layer {layer}"

    def removedir(self, path: str) -> None:
        self.check()
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            if self.base_fs.isdir(path):
                self.mark_deletion(path)
            else:
                raise fs.errors.FileExpected(path)
        elif layer == ADD_LAYER:
            if self.additions_fs.isdir(path):
                self.additions_fs.removedir(path)
                self.mark_deletion(path)
            else:
                raise fs.errors.DirectoryExpected(path)
        elif layer == ROOT_LAYER:
            raise fs.errors.RemoveRootError(path)
        else:
            assert False, f"unknown layer {layer}"

    def setinfo(self, path: str, info: _INFO_DICT) -> None:
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            self.copy_up(path)
            self.additions_fs.setinfo(path, info)
        elif layer == ADD_LAYER:
            self.additions_fs.setinfo(path, info)
        elif layer == ROOT_LAYER:
            pass
        else:
            assert False, f"unknown layer {layer}"

    ############################################################

    def makedirs(
        self,
        path: str,
        permissions: Optional[Permissions] = None,
        recreate: bool = False,
    ) -> SubFS[FS]:
        return FS.makedirs(self, path, permissions=permissions, recreate=recreate)
