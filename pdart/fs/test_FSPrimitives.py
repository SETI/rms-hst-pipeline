import os
import os.path
import shutil
import unittest

from fs.osfs import OSFS
import fs.path
from fs.test import FSTestCases

from pdart.fs.FSPrimAdapter import FSPrimAdapter
from pdart.fs.FSPrimitives import *
from pdart.fs.SubdirVersions import read_subdir_versions_from_directory, \
    write_subdir_versions_to_directory
from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME

if TYPE_CHECKING:
    from typing import Any


class FSPrimitives_TestBase(object):
    """
    This is not a test case, but an abstract base class for a test case.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def setUp(self):
        # type: () -> None
        pass

    def test_root_node(self):
        # type: () -> None
        fs = self.get_fs()
        root = fs.root_node()
        # assert that the root exists...
        self.assertTrue(root)
        # ...and is a directory
        self.assertTrue(fs.is_dir(root))

    def test_is_dir(self):
        # type: () -> None
        fs = self.get_fs()
        root = fs.root_node()
        self.assertTrue(fs.is_dir(root))
        file = fs.add_child_file(root, 'foo')
        self.assertFalse(fs.is_dir(file))

    def test_is_file(self):
        # type: () -> None
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.is_file(root))
        file = fs.add_child_file(root, 'foo')
        self.assertTrue(fs.is_file(file))

    def test_get_dir_children(self):
        # type: () -> None
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.get_dir_children(root))
        file_node = fs.add_child_file(root, 'file')
        dir_node = fs.add_child_dir(root, 'dir')
        expected = {'file': file_node, 'dir': dir_node}
        self.assertEqual(expected, fs.get_dir_children(root))

    def test_get_file_handle(self):
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.get_children(root))
        file_node = fs.add_child_file(root, 'file')
        self.assertTrue(fs.get_file_handle(file_node, 'w'))

    def test_add_child_dir(self):
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.get_children(root))
        dir = fs.add_child_dir(root, 'dir')
        self.assertTrue(fs.is_dir(dir))
        self.assertEqual(dir, fs.get_dir_child(root, 'dir'))
        self.assertEqual('/dir', dir.path)

    def test_add_child_file(self):
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.get_children(root))
        file = fs.add_child_file(root, 'file')
        self.assertTrue(fs.is_file(file))
        self.assertEqual(file, fs.get_dir_child(root, 'file'))
        self.assertEqual('/file', file.path)

    def test_remove_child(self):
        fs = self.get_fs()
        root = fs.root_node()
        dir = fs.add_child_dir(root, 'dir')
        file = fs.add_child_file(root, 'file')
        self.assertEqual({'dir': dir, 'file': file},
                         fs.get_dir_children(root))
        fs.remove_child(root, 'dir')
        self.assertEqual({'file': file},
                         fs.get_dir_children(root))
        fs.remove_child(root, 'file')
        self.assertFalse(fs.get_dir_children(root))

    # The following are defined as abstract.  Their implementations
    # come from mixing with unittest.TestCase.  I can't inherit from
    # TestCase here because then py.test will try to construct and run
    # this abstract class.

    @abc.abstractmethod
    def assertTrue(self, cond, msg=None):
        # type: (Any, object) -> None
        pass

    @abc.abstractmethod
    def assertFalse(self, cond, msg=None):
        # type: (Any, object) -> None
        pass

    @abc.abstractmethod
    def assertEqual(self, lhs, rhs, msg=None):
        # type: (Any, Any, object) -> None
        pass

    # This is also defined in the real TestCase.

    @abc.abstractmethod
    def get_fs(self):
        pass


V1_0 = u'v$1.0'


class V1Primitives(FSPrimitives):
    def __init__(self, root):
        # type: (unicode) -> None
        FSPrimitives.__init__(self)
        self.root = root

    def __str__(self):
        return 'V1Primitives(%r)' % self.root

    def __repr__(self):
        return 'V1Primitives(%r)' % self.root

    def root_node(self):
        # type: () -> Dir_
        return Dir(self, u'/')

    def is_file(self, node):
        # type: (Node_) -> bool
        l, parts, path = self._do_path(node)
        if l is 0:
            return False
        elif l is 1:
            sys_path = fs.path.join(self.root, path.lstrip('/'))
            return os.path.isfile(sys_path)
        elif l in [2, 3]:
            sys_parts = [self.root] + parts[:-1] + [V1_0, parts[-1]]
            sys_path = fs.path.join(*sys_parts)
            return os.path.isfile(sys_path)
        elif l is 4:
            return True
        else:  # l > 4:
            assert False, ('is_file(%r): '
                           'directories in products not current allowed' %
                           path)

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
            v1_sys_path = fs.path.join(self.root, path.lstrip('/'), V1_0)
            osfs = OSFS(v1_sys_path)
            res = {}
            if l is not 3:
                d = read_subdir_versions_from_directory(osfs, u'/')
                for filename in d:
                    res[filename] = Dir(self, fs.path.join(path, filename))
            for filename in osfs.listdir(u'/'):
                assert osfs.isfile(filename)
                res[filename] = File(self, fs.path.join(path, filename))
            return res
        else:
            assert False, ('get_dir_children(%r): '
                           'directories in products not currently allowed' %
                           fs.path.join(path, filename))

    def add_child_dir(self, parent_node, filename):
        # type: (Dir_, unicode) -> Dir_
        l, parts, path = self._do_path(parent_node)
        if l in [0, 1, 2]:
            if l in [1, 2]:
                v1_path = fs.path.join(self.root, path.lstrip('/'), V1_0)
                osfs = OSFS(v1_path)
                d = read_subdir_versions_from_directory(osfs, u'/')
                d[str(filename)] = '1.0'
                write_subdir_versions_to_directory(osfs, u'/', d)

            sys_path = fs.path.join(self.root, path.lstrip('/'), filename)
            os.mkdir(sys_path)
            sys_path = fs.path.join(sys_path, V1_0)
            os.mkdir(sys_path)
            if l is not 2:
                write_subdir_versions_to_directory(OSFS(sys_path), u'/', {})
            return Dir(self, fs.path.join(path, filename))
        else:
            assert False, ('add_child_dir(%r): '
                           'directories in products not currently allowed' %
                           fs.path.join(path, filename))

    def add_child_file(self, parent_node, filename):
        # type: (Dir_, unicode) -> File_
        l, parts, path = self._do_path(parent_node)
        if l is 0:
            sys_path = fs.path.join(self.root, filename)
        elif l in [1, 2, 3]:
            sys_parts = [self.root] + parts + [V1_0, filename]
            sys_path = fs.path.join(*sys_parts)
        else:
            assert False, ('add_child_file(%r): '
                           'directories in products not currently allowed' %
                           fs.path.join(path, filename))
        with open(sys_path, 'w'):
            pass
        return File(self, fs.path.join(path, filename))

    def get_file_handle(self, node, mode):
        # type: (File, str) -> io.IOBase
        l, parts, path = self._do_path(node)
        if l is 0:
            assert False, "get_file_handle(u'/')"
        elif l is 1:
            sys_path = fs.path.join(self.root, parts[0])
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
                sys_parts = [self.root] + parts + [V1_0, filename]
            sys_path = fs.path.join(*sys_parts)
            os.remove(sys_path)
        else:
            # remove dir
            assert l in [0, 1, 2]
            sys_parts = [self.root] + parts + [filename]
            sys_path = fs.path.join(*sys_parts)
            v1_path = fs.path.join(sys_path, V1_0)
            if l is not 2:
                subdir_versions_path = fs.path.join(v1_path,
                                                    SUBDIR_VERSIONS_FILENAME)
                assert os.path.exists(subdir_versions_path), \
                    subdir_versions_path
                os.remove(subdir_versions_path)
            os.rmdir(v1_path)
            os.rmdir(sys_path)

    def _do_path(self, node):
        path = node.path
        parts = fs.path.iteratepath(path)
        return (len(parts), parts, path)


class OSFSPrimitives(FSPrimitives):
    def __init__(self, root):
        # type: (unicode) -> None
        FSPrimitives.__init__(self)
        self.root = root

    def __str__(self):
        return 'OSFSPrimitives(%r)' % self.root

    def __repr__(self):
        return 'OSFSPrimitives(%r)' % self.root

    def _to_sys_path(self, path):
        # type: (unicode) -> unicode
        path_part = path.lstrip('/')
        return os.path.join(self.root, path_part)

    def add_child_dir(self, parent_node, filename):
        # type: (Dir_, unicode) -> Dir_
        path = fs.path.join(parent_node.path, filename)
        sys_path = self._to_sys_path(path)
        os.mkdir(sys_path)
        return Dir(self, path)

    def add_child_file(self, parent_node, filename):
        # type: (Dir_, unicode) -> File_
        path = fs.path.join(parent_node.path, filename)
        sys_path = self._to_sys_path(path)
        with open(sys_path, 'w'):
            pass
        return File(self, path)

    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        dir_path = node.path
        dir_sys_path = self._to_sys_path(dir_path)
        res = dict()
        for filename in os.listdir(dir_sys_path):
            child_path = fs.path.join(dir_path, filename)
            if os.path.isfile(child_path):
                child_node = File(self, child_path)  # type: Node
            else:
                child_node = Dir(self, child_path)
            res[unicode(filename)] = child_node
        return res

    def get_file_handle(self, node, mode):
        # type: (File, str) -> io.IOBase
        sys_path = self._to_sys_path(node.path)
        return cast(io.IOBase,
                    io.open(sys_path, fs.mode.Mode(mode).to_platform_bin()))
        # The cast is due to a bug in the mypy, testing, typeshed
        # environment.

    def is_file(self, node):
        # type: (Node_) -> bool
        sys_path = self._to_sys_path(node.path)
        return os.path.isfile(sys_path)

    def remove_child(self, parent_node, filename):
        # type: (Dir_, unicode) -> None
        child = self.get_dir_child(parent_node, filename)
        sys_path = self._to_sys_path(child.path)
        if self.is_file(child):
            os.remove(sys_path)
        else:
            os.rmdir(sys_path)

    def root_node(self):
        # type: () -> Dir_
        return Dir(self, u'/')


_TMP_DIR = os.path.abspath('tmp_osfs_prims')


class Test_V1Primitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        try:
            os.mkdir(_TMP_DIR)
        except OSError:
            shutil.rmtree(_TMP_DIR)
            os.mkdir(_TMP_DIR)

        self.fs = V1Primitives(_TMP_DIR)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        shutil.rmtree(_TMP_DIR)


class Test_OSFSPrimitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        try:
            os.mkdir(_TMP_DIR)
        except OSError:
            shutil.rmtree(_TMP_DIR)
            os.mkdir(_TMP_DIR)

        self.fs = OSFSPrimitives(_TMP_DIR)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        shutil.rmtree(_TMP_DIR)


class OSFSPrimAdapter(FSPrimAdapter):
    def __init__(self, root_dir):
        FSPrimAdapter.__init__(self, OSFSPrimitives(root_dir))

    def getsyspath(self, path):
        return self.prims._to_sys_path(path)


class Test_OSFSPrimAdapter(FSTestCases, unittest.TestCase):
    def make_fs(self):
        try:
            os.mkdir(_TMP_DIR)
        except OSError:
            shutil.rmtree(_TMP_DIR)
            os.mkdir(_TMP_DIR)
        return OSFSPrimAdapter(_TMP_DIR)
