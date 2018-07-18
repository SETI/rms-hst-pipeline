import abc

import fs.errors
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import io


class Node(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, prims, path):
        # type: (FSPrimitives, unicode) -> None
        self.prims = prims
        self.path = path

    @abc.abstractmethod
    def is_file(self):
        # type: () -> bool
        pass

    def is_dir(self):
        # type: () -> bool
        return not self.is_file()

    def __eq__(self, rhs):
        return self.path == rhs.path


class Dir(Node):
    def __init__(self, prims, path):
        # type: (FSPrimitives, unicode) -> None
        Node.__init__(self, prims, path)

    def is_file(self):
        # type: () -> bool
        return False

    def __repr__(self):
        return 'Dir(%r, %r)' % (self.prims, self.path)


class File(Node):
    def __init__(self, prims, path):
        # type: (FSPrimitives, unicode) -> None
        Node.__init__(self, prims, path)

    def is_file(self):
        # type: () -> bool
        return True

    def __repr__(self):
        return 'File(%r, %r)' % (self.prims, self.path)


if TYPE_CHECKING:
    import io

    Node_ = Node
    Dir_ = Dir
    File_ = File

    # # More efficient
    # Node_ = unicode
    # Dir_ = unicode
    # File_ = unicode


class FSPrimitives(object):
    """
    We model a filesystem with a very simple model that we can reason
    about.  The filesystem is like a tree with two kinds of nodes:
    files and dirs.  The file contains some way to change its
    contents, a handle, and dirs contain a map from strings to other
    nodes.  That's it.

    The root of the filesystem is constrained to be a dir.

    There are only a handful of primitive operations that we need to
    check.

    The laws a filesystem must uphold are few.  See the tests.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def root_node(self):
        # type: () -> Dir_
        pass

    def is_dir(self, node):
        # type: (Node_) -> bool
        return not self.is_file(node)

    @abc.abstractmethod
    def is_file(self, node):
        # type: (Node_) -> bool
        pass

    def get_children(self, node):
        # type: (Node_) -> Dict[unicode, Node_]
        if self.is_file(node):
            raise fs.errors.DirectoryExpected(node.path)
        return self.get_dir_children(cast(Dir, node))

    @abc.abstractmethod
    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        pass

    def get_dir_child(self, parent_node, filename):
        children = self.get_children(parent_node)
        return children[filename]

    def get_handle(self, node, mode):
        # type: (Node_, str) -> io.IOBase
        if self.is_dir(node):
            raise fs.errors.FileExpected(node.path)
        return self.get_file_handle(cast(File, node), mode)

    @abc.abstractmethod
    def get_file_handle(self, node, mode):
        # type: (File, str) -> io.IOBase
        pass

    @abc.abstractmethod
    def add_child_dir(self, parent_node, filename):
        # type: (Dir_, unicode) -> Dir_
        pass

    @abc.abstractmethod
    def add_child_file(self, parent_node, filename):
        # type: (Dir_, unicode) -> File_
        pass

    @abc.abstractmethod
    def remove_child(self, parent_node, filename):
        # type: (Dir_, unicode) -> None
        pass
