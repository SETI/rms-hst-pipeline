import re
from typing import Any, Dict, Optional, cast

import fs.path
from fs.base import FS
from fs.info import Info

from pdart.fs.primitives.DirUtils import lid_to_dir
from pdart.fs.primitives.FSPrimAdapter import FSPrimAdapter
from pdart.fs.primitives.FSPrimitives import Dir, FSPrimitives, File, Node
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LIDVID import LIDVID

_NO_VISIT = "no_visit"


def _is_visit_dir(filename: str) -> bool:
    return re.match(r"^visit_..$", filename) is not None


def _visit_of(filename: str) -> Optional[str]:
    try:
        hf = HstFilename(filename)
        return "visit_" + hf.visit()
    except AssertionError:
        return None


def _union_dicts(*ds: Dict[str, Node]) -> Dict[str, Node]:
    res = {}
    for d in ds:
        res.update(d)
    return res


_IS_DIR: bool = False
_IS_FILE: bool = True


def lidvid_to_dirpath(lidvid: LIDVID) -> str:
    """
    Useful for making PDS4 delivery manifests.
    """
    lid = lidvid.lid()
    dir = lid_to_dir(lid)
    dirpath = _translate_path_to_base_path(dir, _IS_DIR)
    return dirpath


def _translate_path_to_base_path(path: str, is_file_hint: Optional[bool] = None) -> str:
    """
    Translates a LID-based path to a human-preferred DeliverableFS
    path.  For instance:
    <bundle>/<collection>/<product>/<product_file> gets translated to
    <bundle>/<collection>/<visit>/<product_file>.
    """
    parts = fs.path.iteratepath(path)
    l_ = len(parts)
    if l_ in [0, 1, 2]:
        return path
    elif l_ is 3:

        # There are three cases.  First, are we looking at a path for
        # a file or for a directory?

        b, c, p = parts
        if is_file_hint is None:
            is_file_hint = "." in p
        if is_file_hint:
            # It's a path to a collection file, perhaps a label or
            # inventory, and is fine as is.
            return fs.path.abspath(path)
        else:
            # It's a path to a directory.  Does it look like a PDS4
            # product directory?
            v = _visit_of(p)
            if v:
                # It looks like a PDS4 product dir and we'll treat it
                # as such.
                return fs.path.abspath(fs.path.join(b, c, v))
            else:
                # It's not a PDS4 product dir.  This is not expected
                # in our archive, but to make the pyfilesystem testing
                # work, we allow it and put the extra stuff into a
                # special dir.
                return fs.path.abspath(fs.path.join(b, c, _NO_VISIT, p))
    else:
        # l_ > 3.  Let's name the first three parts.
        b, c, p = parts[:3]

        # Does the third part look like a PDS4 product directory or
        # not: does it encode a visit?
        v = _visit_of(p)
        if v:
            # It looks like a PDS4 product dir and we'll treat it as
            # such.
            new_parts = parts[:2] + [v] + parts[3:]
            return fs.path.abspath(fs.path.join(*new_parts))
        else:
            # It does not.  We'll file it in a special directory.
            new_parts = parts[:2] + [_NO_VISIT] + parts[2:]
            return fs.path.abspath(fs.path.join(*new_parts))


class DeliverablePrimitives(FSPrimitives):
    """
    This is a filesystem adapter.  Its interface is LID-based; the
    backing storage is in a human-readable hierarchy which dumps all
    product files into common visit directories under the collection.

    From the outside you see
    <bundle>/<collection>/<product>/<product_file> but the base
    filesystem stores <bundle>/<collection>/<visit>/<product_file>.
    """

    def __init__(self, base_fs: FS) -> None:
        FSPrimitives.__init__(self)
        self.base_fs = base_fs
        self._product_dirs: Dict[str, str] = {}

    def _add_product_dir(self, path: str, base_path: str) -> None:
        assert len(fs.path.iteratepath(path)) == 3
        self._product_dirs[path] = base_path

    def _rm_product_dir(self, path: str) -> None:
        base_path = self._product_dirs[path]
        del self._product_dirs[path]
        # If there are no more in the visit dir, remove the visit dir
        # in the base_fs
        if base_path not in self._product_dirs:
            self.base_fs.removedir(base_path)

    def __str__(self) -> str:
        return f"DeliverablePrimitives({self.base_fs})"

    def __repr__(self) -> str:
        return f"DeliverablePrimitives({self.base_fs!r}"

    def add_child_dir(self, parent_node: Dir, filename: str) -> Dir:
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path_to_base_path(path, _IS_DIR)
        if not self.base_fs.exists(base_path):
            # You might have to insert a dir like _NO_VISIT, so we use
            # makedirs() instead of makedir().
            self.base_fs.makedirs(base_path)

        path_parts = list(fs.path.iteratepath(path))
        if len(path_parts) == 3 and _visit_of(path_parts[2]):
            self._add_product_dir(path, base_path)

        return Dir(self, path)

    def add_child_file(self, parent_node: Dir, filename: str) -> File:
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path_to_base_path(path, _IS_FILE)
        self.base_fs.touch(base_path)
        return File(self, path)

    def _info_to_node(self, parent_dir_path: str, info: Info) -> Node:
        filepath = fs.path.join(parent_dir_path, info.name)
        if info.is_file:
            return File(self, filepath)
        else:
            return Dir(self, filepath)

    def _file_contents(self, path: str) -> Dict[str, Node]:
        """
        Return the contents of the path that are files.  This is only
        called when the path has two parts, so the path is also the
        basepath.
        """
        assert len(fs.path.iteratepath(path)) == 2
        return {
            info.name: self._info_to_node(path, info)
            for info in self.base_fs.scandir(path, namespaces=["basic"])
            if info.is_file
        }

    def _visit_contents(self, path: str) -> Dict[str, Node]:
        """
        Return the contents of the path that are product directories.
        This is only called when the path has two parts, so the path
        is also the basepath.
        """
        assert len(fs.path.iteratepath(path)) == 2
        res: Dict[str, Node] = {}
        for dir in self._product_dirs:
            if fs.path.isbase(path, dir):
                basename = fs.path.basename(dir)
                dir_path = fs.path.join(path, basename)
                res[str(basename)] = Dir(self, dir_path)
        return res

    def _doc_contents(self, path: str) -> Dict[str, Node]:
        b, c = fs.path.iteratepath(path)
        if c == "document":
            return {
                info.name: self._info_to_node(path, info)
                for info in self.base_fs.scandir(path, namespaces=["basic"])
            }
        else:
            return {}

    def _no_visit_contents(self, path: str) -> Dict[str, Node]:
        """
        Return the contents of the path that are directories but not
        product directories.  This is only called when the path has
        two parts, so the path is also the basepath.
        """
        parts = fs.path.iteratepath(path)
        assert len(parts) == 2
        b, c = parts
        new_path = fs.path.join(path, _NO_VISIT)
        if not self.base_fs.isdir(new_path):
            return {}

        return {
            info.name: self._info_to_node(path, info)
            for info in self.base_fs.scandir(new_path, namespaces=["basic"])
        }

    def _product_dir_contents(
        self, path: str, visit_dir_name: str, product_dir_name: str
    ) -> Dict[str, Node]:
        dir_path_parts = fs.path.iteratepath(path)
        assert len(dir_path_parts) == 3
        b, c, p = dir_path_parts
        base_dir_to_list = fs.path.join(b, c, visit_dir_name)
        res = {}

        for info in self.base_fs.scandir(base_dir_to_list):
            filename = info.name
            if filename[:9] == product_dir_name:
                res[str(filename)] = self._info_to_node(path, info)

        return res

    def get_dir_children(self, node: Dir) -> Dict[str, Node]:
        res: Dict[str, Node] = {}
        path = node.path
        path_parts = list(fs.path.iteratepath(path))
        l_ = len(path_parts)
        base_path = _translate_path_to_base_path(path, _IS_DIR)
        if l_ not in [2, 3]:
            for filename in self.base_fs.listdir(base_path):
                child_path = fs.path.join(path, filename)
                base_child_path = fs.path.join(base_path, filename)
                if self.base_fs.isfile(base_child_path):
                    res[str(filename)] = File(self, child_path)
                else:
                    res[str(filename)] = Dir(self, child_path)
            return res
        elif l_ is 2:
            # note that path == base_path for l_ == 2
            return _union_dicts(
                self._file_contents(path),
                self._visit_contents(path),
                self._no_visit_contents(path),
                self._doc_contents(path),
            )
        else:
            assert l_ is 3
            product_dir_name = path_parts[2]
            v = _visit_of(product_dir_name)
            if v:
                # Treat it as a product dir; assume it exists.
                # Go over the contents of the visit dir
                return self._product_dir_contents(path, v, product_dir_name)
            else:
                # treat is as a generic dir
                for filename in self.base_fs.listdir(base_path):
                    child_path = fs.path.join(path, filename)
                    base_child_path = fs.path.join(base_path, filename)
                    if self.base_fs.isfile(base_child_path):
                        res[str(filename)] = File(self, child_path)
                    else:
                        res[str(filename)] = Dir(self, child_path)
            return res

    def get_file_handle(self, node: File, mode: str) -> Any:
        base_path = _translate_path_to_base_path(node.path, _IS_FILE)
        return self.base_fs.openbin(base_path, mode)

    def is_file(self, node: Node) -> bool:
        base_path = _translate_path_to_base_path(node.path, _IS_FILE)
        return self.base_fs.isfile(base_path)

    def _remove_no_visit_dir_if_empty(self, path: str) -> None:
        parts = fs.path.iteratepath(path)
        assert len(parts) == 2
        no_visit_base_dir = fs.path.join(path, _NO_VISIT)
        if not self.base_fs.listdir(no_visit_base_dir):
            self.base_fs.removedir(no_visit_base_dir)

    def remove_child(self, parent_node: Dir, filename: str) -> None:
        base_fs = self.base_fs
        path = fs.path.join(parent_node.path, filename)
        parts = list(fs.path.iteratepath(path))
        l_ = len(parts)
        if l_ is 3:
            if _visit_of(parts[2]):
                # it's of the form /bundle/collection/product
                self._rm_product_dir(path)
            else:
                # it's just a /b/c/p
                base_path = _translate_path_to_base_path(path, _IS_FILE)
                if base_fs.isfile(base_path):
                    base_fs.remove(base_path)
                else:
                    # it's a non-PDS4 directory
                    base_path = _translate_path_to_base_path(path, _IS_DIR)
                    base_fs.removedir(base_path)
                    self._remove_no_visit_dir_if_empty(parent_node.path)
        else:
            base_path = _translate_path_to_base_path(path, _IS_FILE)
            if base_fs.isfile(base_path):
                base_fs.remove(base_path)
            else:
                base_fs.removedir(base_path)

    def root_node(self) -> Dir:
        return Dir(self, "/")

    def _to_sys_path(self, path: str) -> str:
        base_path = _translate_path_to_base_path(path, _IS_FILE)
        return self.base_fs.getsyspath(base_path)


class DeliverableFS(FSPrimAdapter):
    def __init__(self, base_fs: FS) -> None:
        FSPrimAdapter.__init__(self, DeliverablePrimitives(base_fs))

    def getsyspath(self, path: str) -> str:
        return cast(DeliverablePrimitives, self.prims)._to_sys_path(path)
