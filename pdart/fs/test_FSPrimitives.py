import os
import os.path
import shutil
import unittest

from pdart.fs.FSPrimitives import *

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
        self.assertTrue(fs.is_dir_prim(root))

    def test_is_dir_prim(self):
        # type: () -> None
        fs = self.get_fs()
        root = fs.root_node()
        self.assertTrue(fs.is_dir_prim(root))
        file = fs.add_child_file(root, 'foo')
        self.assertFalse(fs.is_dir_prim(file))

    def test_is_file_prim(self):
        # type: () -> None
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.is_file_prim(root))
        file = fs.add_child_file(root, 'foo')
        self.assertTrue(fs.is_file_prim(file))

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
        self.assertTrue(fs.is_dir_prim(dir))
        self.assertEqual(dir, fs.get_dir_child(root, 'dir'))

    def test_add_child_file(self):
        fs = self.get_fs()
        root = fs.root_node()
        self.assertFalse(fs.get_children(root))
        file = fs.add_child_file(root, 'file')
        self.assertTrue(fs.is_file_prim(file))
        self.assertEqual(file, fs.get_dir_child(root, 'file'))

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


class OSFSPrimitives(FSPrimitives):
    def __init__(self, root):
        # type: (unicode) -> None
        FSPrimitives.__init__(self)
        self.root = root

    def add_child_dir(self, parent_node, name):
        # type: (Dir_, unicode) -> Dir_
        path = os.path.join(parent_node.name, name)
        os.mkdir(path)
        return Dir(path)

    def add_child_file(self, parent_node, name):
        # type: (Dir_, unicode) -> File_
        path = os.path.join(parent_node.name, name)
        with open(path, 'w'):
            pass
        return File(path)

    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        dir_path = node.name
        res = dict()
        for name in os.listdir(dir_path):
            child_path = os.path.join(dir_path, name)
            if os.path.isfile(child_path):
                child_node = File(child_path)  # type: Node
            else:
                child_node = Dir(child_path)
            res[unicode(name)] = child_node
        return res

    def get_file_handle(self, node, mode):
        # type: (File, str) -> Any
        return open(node.name, mode)

    def is_file_prim(self, node):
        # type: (Node_) -> bool
        return os.path.isfile(node.name)

    def remove_child(self, parent_node, name):
        # type: (Dir_, unicode) -> None
        child = self.get_dir_child(parent_node, name)
        if self.is_file_prim(child):
            os.remove(child.name)
        else:
            os.rmdir(child.name)

    def root_node(self):
        # type: () -> Dir_
        return Dir(self.root)


_TMP_DIR = 'tmp_osfs_prims'


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
