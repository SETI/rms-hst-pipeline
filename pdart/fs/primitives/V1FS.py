import io
import os
from typing import Dict, List, Tuple, cast

import fs.errors
from fs.osfs import OSFS

from pdart.fs.primitives.FSPrimAdapter import FSPrimAdapter
from pdart.fs.primitives.FSPrimitives import Dir, FSPrimitives, File, Node
from pdart.fs.primitives.SubdirVersions import (
    read_subdir_versions_from_directory,
    write_subdir_versions_to_directory,
)
from pdart.fs.primitives.VersionedFS import SUBDIR_VERSIONS_FILENAME

_V1_0: str = "v$1.0"


class V1Primitives(FSPrimitives):
    """
    A primitives wrapper around an OS directory.  Write to the wrapper
    as if you're writing a LID-based hierarchy (i.e.,
    /<bundle>/<collection>/<product>/<product-files>), and it will
    appear in the OS directory as a LIDVID-based hierarchy with all
    written files at version 1.0.
    """

    def __init__(self, root: str) -> None:
        FSPrimitives.__init__(self)
        self.root = root

    def __str__(self) -> str:
        return f"V1Primitives({self.root})"

    def __repr__(self) -> str:
        return f"V1Primitives({self.root!r})"

    def _to_sys_path(self, path: str) -> str:
        """
        If the path is a file at the top level, return it relative to
        the root.  In all other cases, insert a version number after
        the last directory in the path.

        This method is not elegant.
        """
        path = fs.path.abspath(path)
        file = File(self, path)
        if self.is_file(file):
            (parent_dir, filepath) = fs.path.split(path)
            if parent_dir == "/":
                res = fs.path.join(self.root, filepath)
            else:
                res = fs.path.join(self.root, parent_dir.lstrip("/"), _V1_0, filepath)
            return res
        else:
            # it's a dir.
            return fs.path.join(self.root, path.lstrip("/"), _V1_0)

    def root_node(self) -> Dir:
        return Dir(self, "/")

    def too_deep(self, func: str, path: str) -> None:
        assert False, f"{func}({path!r}): directories in products not currently allowed"

    def is_file(self, node: Node) -> bool:
        l, parts, path = self._do_path(node)
        if l is 0:
            return False
        elif l is 1:
            sys_path = fs.path.join(self.root, path.lstrip("/"))
            return os.path.isfile(sys_path)
        elif l in [2, 3]:
            sys_parts = [self.root] + parts[:-1] + [_V1_0, parts[-1]]
            sys_path = fs.path.join(*sys_parts)
            return os.path.isfile(sys_path)
        elif l is 4:
            return True
        else:  # l > 4:
            self.too_deep("is_file", path)
            raise Exception("V1FS.is_file: too deep")

    def get_dir_children(self, node: Dir) -> Dict[str, Node]:
        l, parts, path = self._do_path(node)
        if l is 0:
            res = {}
            for filename in os.listdir(self.root):
                child_path = fs.path.join(path, filename)
                child_syspath = fs.path.join(self.root, filename)
                if os.path.isfile(child_syspath):
                    child_node: Node = File(self, child_path)
                else:
                    child_node = Dir(self, child_path)
                res[str(filename)] = child_node
            return res
        elif l in [1, 2, 3]:
            v1_sys_path = fs.path.join(self.root, path.lstrip("/"), _V1_0)
            osfs = OSFS(v1_sys_path)
            res = {}
            if l is not 3:
                d = read_subdir_versions_from_directory(osfs, "/")
                for filename in d:
                    res[filename] = Dir(self, fs.path.join(path, filename))
            for filename in osfs.listdir("/"):
                assert osfs.isfile(filename)
                if not filename == SUBDIR_VERSIONS_FILENAME:
                    res[filename] = File(self, fs.path.join(path, filename))
            return res
        else:
            self.too_deep("get_dir_children", f"{fs.path.join(path, filename)!r}")
            assert False

    def add_child_dir(self, parent_node: Dir, filename: str) -> Dir:
        l, parts, path = self._do_path(parent_node)
        if l in [0, 1, 2]:
            if l in [1, 2]:
                v1_path = fs.path.join(self.root, path.lstrip("/"), _V1_0)
                osfs = OSFS(v1_path)
                d = read_subdir_versions_from_directory(osfs, "/")
                d[str(filename)] = "1.0"
                write_subdir_versions_to_directory(osfs, "/", d)

            sys_path = fs.path.join(self.root, path.lstrip("/"), filename)
            os.mkdir(sys_path)
            sys_path = fs.path.join(sys_path, _V1_0)
            os.mkdir(sys_path)
            if l is not 2:
                write_subdir_versions_to_directory(OSFS(sys_path), "/", {})
            return Dir(self, fs.path.join(path, filename))
        else:
            self.too_deep("add_child_dir", fs.path.join(path, filename))
            assert False

    def add_child_file(self, parent_node: Dir, filename: str) -> File:
        l, parts, path = self._do_path(parent_node)
        if l is 0:
            sys_path = fs.path.join(self.root, filename)
        elif l in [1, 2, 3]:
            sys_parts = [self.root] + parts + [_V1_0, filename]
            sys_path = fs.path.join(*sys_parts)
        else:
            self.too_deep("add_child_file", fs.path.join(path, filename))

        with open(sys_path, "w"):
            pass
        return File(self, fs.path.join(path, filename))

    def get_file_handle(self, node: File, mode: str) -> io.IOBase:
        l, parts, path = self._do_path(node)
        if l is 0:
            assert False, "get_file_handle(u'/')"
        elif l is 1:
            sys_path = fs.path.join(self.root, *parts)
            return cast(
                io.IOBase, io.open(sys_path, fs.mode.Mode(mode).to_platform_bin())
            )
        elif l in [2, 3, 4]:
            sys_parts = [self.root] + parts[:-1] + [_V1_0, parts[-1]]
            sys_path = fs.path.join(*sys_parts)
            return cast(
                io.IOBase, io.open(sys_path, fs.mode.Mode(mode).to_platform_bin())
            )
        else:
            assert False, f"get_file_handle({path!r})"

    def remove_child(self, parent_node: Dir, filename: str) -> None:
        l, parts, path = self._do_path(parent_node)
        child = self.get_dir_child(parent_node, filename)
        if self.is_file(child):
            # remove file
            if l is 0:
                sys_parts = [self.root, filename]
            else:
                sys_parts = [self.root] + parts + [_V1_0, filename]
            sys_path = fs.path.join(*sys_parts)
            os.remove(sys_path)
        else:
            # remove dir
            assert l in [0, 1, 2]
            sys_parts = [self.root] + parts + [filename]
            sys_path = fs.path.join(*sys_parts)
            v1_path = fs.path.join(sys_path, _V1_0)
            if l is not 2:
                subdir_versions_path = fs.path.join(v1_path, SUBDIR_VERSIONS_FILENAME)
                assert os.path.exists(subdir_versions_path), subdir_versions_path
                os.remove(subdir_versions_path)
            if l in [1, 2]:
                v1_sys_path = fs.path.join(self.root, path.lstrip("/"), _V1_0)
                osfs = OSFS(v1_sys_path)
                d = read_subdir_versions_from_directory(osfs, "/")
                del d[str(filename)]
                write_subdir_versions_to_directory(osfs, "/", d)
            os.rmdir(v1_path)
            os.rmdir(sys_path)

    def _do_path(self, node: Node) -> Tuple[int, List[str], str]:
        """
        Split the node's path and return a triple of the length of the path,
        the parts, and the path itself.
        """
        path = node.path
        parts = fs.path.iteratepath(path)
        return (len(parts), parts, path)


class V1FS(FSPrimAdapter):
    """
    A pyfilesystems wrapper around an OS directory.  Write to the
    wrapper as if you're writing a LID-based hierarchy (i.e.,
    /<bundle>/<collection>/<product>/<product-files>), and it will
    appear in the OS directory as a LIDVID-based hierarchy with all
    written files at version 1.0.
    """

    def __init__(self, root_dir: str) -> None:
        FSPrimAdapter.__init__(self, V1Primitives(root_dir))

    def getsyspath(self, path: str) -> str:
        res = cast(V1Primitives, self.prims)._to_sys_path(path)
        if res:
            return res
        else:
            raise fs.errors.NoSysPath(path)
