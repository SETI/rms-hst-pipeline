import re

import fs.path
from typing import TYPE_CHECKING

from pdart.fs.FSPrimAdapter import FSPrimAdapter
from pdart.fs.FSPrimitives import Dir, FSPrimitives, File
from pdart.pds4.HstFilename import HstFilename

if TYPE_CHECKING:
    import io
    from fs.base import FS
    from fs.info import Info
    from pdart.fs.FSPrimitives import Dir_, File_, Node_

_NO_VISIT = u'no$visit'


def _is_visit_dir(filename):
    return re.match(r'^visit_..$', filename) is not None


def _visit_of(filename):
    # type: (unicode) -> unicode
    try:
        hf = HstFilename(filename)
        return 'visit_' + hf.visit()
    except AssertionError:
        return None


def _union_dicts(a, b, c):
    res = {}
    res.update(a)
    res.update(b)
    res.update(c)
    return res


_IS_DIR = False  # type: bool
_IS_FILE = True  # type: bool


def _translate_path_to_base_path(path, is_file_hint=None):
    # type: (unicode, bool) -> unicode
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
        # There are three cases.  First, are we looking at a file or a
        # directory?
        if is_file_hint is None:
            is_file_hint = '.' in parts[2]
        if is_file_hint:
            # It's fine as is.
            return path
        else:
            # Does it look like a PDS4 product directory or not?
            b, c, p = parts
            v = _visit_of(p)
            if v:
                # It looks like a PDS4 product dir and we'll treat it
                # as such.
                return fs.path.join(b, c, v)
            else:
                # It's not a PDS4 product dir.  This is not expected
                # in our archive, but to make the pyfilesystem testing
                # work, we allow it and put the extra stuff into a
                # special dir.
                return fs.path.join(b, c, _NO_VISIT, p)
    else:
        # Does the third part look like a PDS4 product directory or not?
        v = _visit_of(parts[2])
        if v:
            # It looks like a PDS4 product dir and we'll treat it as
            # such.
            new_parts = parts[:2] + [v] + parts[3:]
            return fs.path.join(*new_parts)
        else:
            # It does not.
            new_parts = parts[:2] + [_NO_VISIT] + parts[2:]
            return fs.path.join(*new_parts)


class DeliverablePrimitives(FSPrimitives):
    """
    This is a filesystem adapter.  Its interface is LID-based; the
    backing storage is in a human-readable hierarchy which dumps all
    product files into common visit directories under the collection.

    From the outside you see
    <bundle>/<collection>/<product>/<product_file> but the base
    filesystem stores <bundle>/<collection>/<visit>/<product_file>.
    """

    def __init__(self, base_fs):
        # type: (FS) -> None
        FSPrimitives.__init__(self)
        self.base_fs = base_fs
        self.implicit_dirs = {}  # type: Dict[unicode, unicode]

    def _add_implicit_dir(self, path, base_path):
        # type: (unicode, unicode) -> None
        self.implicit_dirs[path] = base_path

    def _rm_implicit_dir(self, path):
        # type: (unicode) -> None
        base_path = self.implicit_dirs[path]
        del self.implicit_dirs[path]
        # If there are no more in the visit dir, remove the visit dir
        # in the base_fs
        if base_path not in self.implicit_dirs:
            self.base_fs.removedir(base_path)

    def __str__(self):
        # type: () -> str
        return 'DeliverablePrimitives(%r)' % self.base_fs

    def __repr__(self):
        # type: () -> str
        return 'DeliverablePrimitives(%r)' % self.base_fs

    def add_child_dir(self, parent_node, filename):
        # type: (Dir_, unicode) -> Dir_
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path_to_base_path(path, _IS_DIR)
        if not self.base_fs.exists(base_path):
            # You might have to insert a dir like _NO_VISIT, so we use
            # makedirs() instead of makedir().
            self.base_fs.makedirs(base_path)

        path_parts = list(fs.path.iteratepath(path))
        if len(path_parts) == 3 and _visit_of(path_parts[2]):
            self._add_implicit_dir(path, base_path)

        return Dir(self, path)

    def add_child_file(self, parent_node, filename):
        # type: (Dir_, unicode) -> File_
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path_to_base_path(path, _IS_FILE)
        self.base_fs.touch(base_path)
        return File(self, path)

    def _file_contents(self, path):
        # type: (unicode) -> Dict[unicode, Node_]
        return {info.name: File(self, fs.path.join(path, info.name))
                for info
                in self.base_fs.scandir(path, namespaces=['basic'])
                if info.is_file}

    def _visit_contents(self, path):
        # type: (unicode) -> Dict[unicode, Node_]
        res = {}  # type: Dict[unicode, Node_]
        for dir in self.implicit_dirs:
            basename = fs.path.basename(dir)
            dir_path = fs.path.join(path, basename)
            res[unicode(basename)] = Dir(self, dir_path)
        return res

    def _no_visit_contents(self, path):
        # type: (unicode) -> Dict[unicode, Node_]
        new_path = fs.path.join(path, _NO_VISIT)
        if not self.base_fs.isdir(new_path):
            return {}

        def info_to_node(info):
            # type: (Info) -> Node_
            filepath = fs.path.join(path, info.name)
            if info.is_file:
                return File(self, filepath)
            else:
                return Dir(self, filepath)

        return {info.name: info_to_node(info)
                for info
                in self.base_fs.scandir(new_path, namespaces=['basic'])}

    def _product_dir_contents(self, path, visit_dir_name, product_dir_name):
        # type: (unicode, unicode, unicode) -> Dict[unicode, Node_]
        dir_path_parts = fs.path.iteratepath(path)
        assert len(dir_path_parts) == 3
        b, c, p = dir_path_parts
        base_dir_to_list = fs.path.join(b, c, visit_dir_name)
        res = {}

        def info_to_node(info):
            # type: (Info) -> Node_
            filepath = fs.path.join(path, info.name)
            if info.is_file:
                return File(self, filepath)
            else:
                return Dir(self, filepath)

        for info in self.base_fs.scandir(base_dir_to_list):
            filename = info.name
            if filename[:9] == product_dir_name:
                res[unicode(filename)] = info_to_node(info)

        return res

    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        res = {}  # type: Dict[unicode, Node_]
        path = node.path
        path_parts = list(fs.path.iteratepath(path))
        l_ = len(path_parts)
        base_path = _translate_path_to_base_path(path, _IS_DIR)
        if l_ not in [2, 3]:
            for filename in self.base_fs.listdir(base_path):
                child_path = fs.path.join(path, filename)
                base_child_path = fs.path.join(base_path, filename)
                if self.base_fs.isfile(base_child_path):
                    res[unicode(filename)] = File(self, child_path)
                else:
                    res[unicode(filename)] = Dir(self, child_path)
            return res
        elif l_ is 2:
            # note that path == base_path for l_ == 2
            return _union_dicts(self._file_contents(path),
                                self._visit_contents(path),
                                self._no_visit_contents(path))
        elif l_ is 3:
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
                        res[unicode(filename)] = File(self, child_path)
                    else:
                        res[unicode(filename)] = Dir(self, child_path)
            return res

    def get_file_handle(self, node, mode):
        # type: (File_, str) -> io.IOBase
        base_path = _translate_path_to_base_path(node.path, _IS_FILE)
        return self.base_fs.openbin(base_path, mode)

    def is_file(self, node):
        # type: (Node_) -> bool
        base_path = _translate_path_to_base_path(node.path, _IS_FILE)
        return self.base_fs.isfile(base_path)

    def _remove_no_visit_dir_if_empty(self, path):
        # type: (unicode) -> None
        parts = fs.path.iteratepath(path)
        assert len(parts) == 2
        no_visit_base_dir = fs.path.join(path, _NO_VISIT)
        if not self.base_fs.listdir(no_visit_base_dir):
            self.base_fs.removedir(no_visit_base_dir)

    def remove_child(self, parent_node, filename):
        # type: (Dir_, unicode) -> None
        base_fs = self.base_fs
        path = fs.path.join(parent_node.path, filename)
        parts = list(fs.path.iteratepath(path))
        l_ = len(parts)
        if l_ is 3:
            if _visit_of(parts[2]):
                # it's of the form /bundle/collection/product
                self._rm_implicit_dir(path)
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

    def root_node(self):
        # type: () -> Dir_
        return Dir(self, u'/')

    def _to_sys_path(self, path):
        # type: (unicode) -> unicode
        base_path = _translate_path_to_base_path(path, _IS_FILE)
        return self.base_fs.getsyspath(base_path)


class DeliverableFS(FSPrimAdapter):
    def __init__(self, base_fs):
        FSPrimAdapter.__init__(self, DeliverablePrimitives(base_fs))

    def getsyspath(self, path):
        return self.prims._to_sys_path(path)
