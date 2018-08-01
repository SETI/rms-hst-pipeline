import io
import os
from typing import TYPE_CHECKING

import fs.errors
from fs.osfs import OSFS

from pdart.fs.FSPrimAdapter import FSPrimAdapter
from pdart.fs.FSPrimitives import *
from pdart.fs.SubdirVersions import read_subdir_versions_from_directory, \
    write_subdir_versions_to_directory
from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME

if TYPE_CHECKING:
    from typing import List, Tuple

_V1_0 = u'v$1.0'


class V1Primitives(FSPrimitives):
    """
    A primitives wrapper around an OS directory.  Write to the wrapper
    as if you're writing a LID-based hierarchy (i.e.,
    /<bundle>/<collection>/<product>/<product-files>), and it will
    appear in the OS directory as a LIDVID-based hierarchy with all
    written files at version 1.0.
    """

    def __init__(self, root):
        # type: (unicode) -> None
        FSPrimitives.__init__(self)
        self.root = root

    def __str__(self):
        return 'V1Primitives(%r)' % self.root

    def __repr__(self):
        return 'V1Primitives(%r)' % self.root

    def _to_sys_path(self, path):
        # type: (unicode) -> unicode
        file = File(self, path)
        if self.is_file(file):
            return fs.path.join(self.root, path.lstrip('/'))
        else:
            # it's a dir.
            return fs.path.join(self.root, path.lstrip('/'), _V1_0)

    def root_node(self):
        # type: () -> Dir_
        return Dir(self, u'/')

    def too_deep(self, func, path):
        assert False, \
            '%s(%r): directories in products not currently allowed' % (func,
                                                                       path)

    def is_file(self, node):
        # type: (Node_) -> bool
        l, parts, path = self._do_path(node)
        if l is 0:
            return False
        elif l is 1:
            sys_path = fs.path.join(self.root, path.lstrip('/'))
            return os.path.isfile(sys_path)
        elif l in [2, 3]:
            sys_parts = [self.root] + parts[:-1] + [_V1_0, parts[-1]]
            sys_path = fs.path.join(*sys_parts)
            return os.path.isfile(sys_path)
        elif l is 4:
            return True
        else:  # l > 4:
            self.too_deep('is_file', path)

    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        l, parts, path = self._do_path(node)
        if l is 0:
            res = {}
            for filename in os.listdir(self.root):
                child_path = fs.path.join(path, filename)
                child_syspath = fs.path.join(self.root, filename)
                if os.path.isfile(child_syspath):
                    child_node = File(self, child_path)  # type: Node
                else:
                    child_node = Dir(self, child_path)
                res[unicode(filename)] = child_node
            return res
        elif l in [1, 2, 3]:
            v1_sys_path = fs.path.join(self.root, path.lstrip('/'), _V1_0)
            osfs = OSFS(v1_sys_path)
            res = {}
            if l is not 3:
                d = read_subdir_versions_from_directory(osfs, u'/')
                for filename in d:
                    res[filename] = Dir(self, fs.path.join(path, filename))
            for filename in osfs.listdir(u'/'):
                assert osfs.isfile(filename)
                if not filename == SUBDIR_VERSIONS_FILENAME:
                    res[filename] = File(self, fs.path.join(path, filename))
            return res
        else:
            self.too_deep('get_dir_children(%r)', fs.path.join(path, filename))

    def add_child_dir(self, parent_node, filename):
        # type: (Dir_, unicode) -> Dir_
        l, parts, path = self._do_path(parent_node)
        if l in [0, 1, 2]:
            if l in [1, 2]:
                v1_path = fs.path.join(self.root, path.lstrip('/'), _V1_0)
                osfs = OSFS(v1_path)
                d = read_subdir_versions_from_directory(osfs, u'/')
                d[str(filename)] = '1.0'
                write_subdir_versions_to_directory(osfs, u'/', d)

            sys_path = fs.path.join(self.root, path.lstrip('/'), filename)
            os.mkdir(sys_path)
            sys_path = fs.path.join(sys_path, _V1_0)
            os.mkdir(sys_path)
            if l is not 2:
                write_subdir_versions_to_directory(OSFS(sys_path), u'/', {})
            return Dir(self, fs.path.join(path, filename))
        else:
            self.too_deep('add_child_dir', fs.path.join(path, filename))

    def add_child_file(self, parent_node, filename):
        # type: (Dir_, unicode) -> File_
        l, parts, path = self._do_path(parent_node)
        if l is 0:
            sys_path = fs.path.join(self.root, filename)
        elif l in [1, 2, 3]:
            sys_parts = [self.root] + parts + [_V1_0, filename]
            sys_path = fs.path.join(*sys_parts)
        else:
            self.too_deep('add_child_file', fs.path.join(path, filename))

        with open(sys_path, 'w'):
            pass
        return File(self, fs.path.join(path, filename))

    def get_file_handle(self, node, mode):
        # type: (File, str) -> io.IOBase
        l, parts, path = self._do_path(node)
        if l is 0:
            assert False, "get_file_handle(u'/')"
        elif l is 1:
            sys_path = fs.path.join(self.root, *parts)
            return cast(io.IOBase,
                        io.open(sys_path,
                                fs.mode.Mode(mode).to_platform_bin()))
        elif l in [2, 3, 4]:
            sys_parts = [self.root] + parts[:-1] + [_V1_0, parts[-1]]
            sys_path = fs.path.join(*sys_parts)
            return cast(io.IOBase,
                        io.open(sys_path,
                                fs.mode.Mode(mode).to_platform_bin()))
        else:
            assert False, 'get_file_handle(%r)' % path

    def remove_child(self, parent_node, filename):
        # type: (Dir_, unicode) -> None
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
                subdir_versions_path = fs.path.join(v1_path,
                                                    SUBDIR_VERSIONS_FILENAME)
                assert os.path.exists(subdir_versions_path), \
                    subdir_versions_path
                os.remove(subdir_versions_path)
            if l in [1, 2]:
                v1_sys_path = fs.path.join(self.root, path.lstrip('/'), _V1_0)
                osfs = OSFS(v1_sys_path)
                d = read_subdir_versions_from_directory(osfs, u'/')
                del d[str(filename)]
                write_subdir_versions_to_directory(osfs, u'/', d)
            os.rmdir(v1_path)
            os.rmdir(sys_path)

    def _do_path(self, node):
        # type: (Node_) -> Tuple[int, List[unicode], unicode]
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

    def __init__(self, root_dir):
        FSPrimAdapter.__init__(self, V1Primitives(root_dir))

    def getsyspath(self, path):
        res = self.prims._to_sys_path(path)
        if res:
            return res
        else:
            raise fs.errors.NoSysPath(path)
