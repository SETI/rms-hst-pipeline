import fs.path
from typing import TYPE_CHECKING

from pdart.fs.FSPrimAdapter import FSPrimAdapter
from pdart.fs.FSPrimitives import Dir, FSPrimitives, File
from pdart.pds4.HstFilename import HstFilename

if TYPE_CHECKING:
    import io
    from fs.base import FS
    from pdart.fs.FSPrimitives import Dir_, File_, Node_


def _visit_of(filename):
    # type: (unicode) -> unicode
    try:
        hf = HstFilename(filename)
        return 'visit_' + hf.visit()
    except AssertionError:
        return filename


_IS_DIR = False  # type: bool
_IS_FILE = True  # type: bool


def _translate_path(path, is_file=None):
    # type: (unicode, bool) -> unicode
    parts = fs.path.iteratepath(path)
    l_ = len(parts)
    if l_ in [0, 1, 2]:
        return path
    elif l_ is 3:
        b, c, pf = parts
        if is_file is None:
            is_file = '.' in pf
        if is_file:
            # assume it's a file
            return fs.path.join(b, c, pf)
        else:
            # assume it's a dir; here, a product name
            v = _visit_of(pf)
            return fs.path.join(b, c, v)
    elif l_ is 4:
        b, c, p, f = parts
        v = _visit_of(f)
        if v == f:
            return fs.path.join(b, c, p, f)
        else:
            return fs.path.join(b, c, v, f)
    else:
        assert False, ('_translate_path(%r) not implemented (too long)' %
                       path)


class UsersPrimitives(FSPrimitives):
    """
    Maps a filesystem organized by LIDs (not LIDVIDs; it's a single
    version) into a filesystem organized according to what users want.

    /<bundle>/<collection>/visit_<visit_number>/filename
    """

    def __init__(self, base_fs):
        # type: (FS) -> None
        FSPrimitives.__init__(self)
        self.base_fs = base_fs

    def __str__(self):
        # type: () -> str
        return 'UsersPrimitives(%r)' % self.base_fs

    def __repr__(self):
        # type: () -> str
        return 'UsersPrimitives(%r)' % self.base_fs

    def add_child_dir(self, parent_node, filename):
        # type: (Dir_, unicode) -> Dir_
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path(path, _IS_DIR)
        self.base_fs.makedir(base_path)
        return Dir(self, path)

    def add_child_file(self, parent_node, filename):
        # type: (Dir_, unicode) -> File_
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path(path, _IS_FILE)
        self.base_fs.touch(base_path)
        return File(self, path)

    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        path = node.path
        base_path = _translate_path(path, _IS_DIR)
        base_parts = fs.path.iteratepath(base_path)
        l_ = len(base_parts)
        if l_ in [0, 1, 2, 3, 4]:
            res = {}  # type: Dict[unicode, Node_]
            base_fs = self.base_fs
            for filename in base_fs.listdir(base_path):
                child_path = fs.path.join(base_path, filename)
                if base_fs.isfile(child_path):
                    res[unicode(filename)] = File(self, child_path)
                else:
                    res[unicode(filename)] = Dir(self, child_path)
            return res
        else:
            assert False, ('get_dir_children(%r): not implemented (too long)' %
                           path)

    def get_file_handle(self, node, mode):
        # type: (File_, str) -> io.IOBase
        base_path = _translate_path(node.path, _IS_FILE)
        return self.base_fs.openbin(base_path, mode)

    def is_file(self, node):
        # type: (Node_) -> bool
        base_path = _translate_path(node.path, _IS_FILE)
        return self.base_fs.isfile(base_path)

    def remove_child(self, parent_node, filename):
        # type: (Dir_, unicode) -> None
        path = fs.path.join(parent_node.path, filename)
        base_path = _translate_path(path, _IS_FILE)
        base_fs = self.base_fs
        if base_fs.isfile(base_path):
            base_fs.remove(base_path)
        else:
            base_fs.removedir(base_path)

    def root_node(self):
        # type: () -> Dir_
        return Dir(self, u'/')

    def _to_sys_path(self, path):
        # type: (unicode) -> unicode
        base_path = _translate_path(path, _IS_FILE)
        return self.base_fs.getsyspath(base_path)


class UsersFS(FSPrimAdapter):
    def __init__(self, base_fs):
        FSPrimAdapter.__init__(self, UsersPrimitives(base_fs))

    def getsyspath(self, path):
        return self.prims._to_sys_path(path)
