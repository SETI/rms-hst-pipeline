import abc
import fs.errors
import io

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    pass


class Node(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        # type: (unicode) -> None
        self.name = name

    @abc.abstractmethod
    def is_file(self):
        # type: () -> bool
        pass

    def is_dir(self):
        # type: () -> bool
        return not self.is_file()

    def __eq__(self, rhs):
        return self.name == rhs.name


class Dir(Node):
    def __init__(self, name):
        # type: (unicode) -> None
        Node.__init__(self, name)

    def is_file(self):
        # type: () -> bool
        return False


class File(Node):
    def __init__(self, name):
        # type: (unicode) -> None
        Node.__init__(self, name)

    def is_file(self):
        # type: () -> bool
        return True


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

    def is_dir_prim(self, node):
        # type: (Node_) -> bool
        return not self.is_file_prim(node)

    @abc.abstractmethod
    def is_file_prim(self, node):
        # type: (Node_) -> bool
        pass

    def get_children(self, node):
        # type: (Node_) -> Dict[unicode, Node_]
        if self.is_file_prim(node):
            raise fs.errors.DirectoryExpected(node.name)
        return self.get_dir_children(cast(Dir, node))

    @abc.abstractmethod
    def get_dir_children(self, node):
        # type: (Dir_) -> Dict[unicode, Node_]
        pass

    def get_dir_child(self, parent_node, name):
        return self.get_children(parent_node)[name]

    def get_handle(self, node, mode):
        # type: (Node_, str) -> io.IOBase
        if self.is_dir_prim(node):
            raise fs.errors.FileExpected(node.name)
        return self.get_file_handle(cast(File, node), mode)

    @abc.abstractmethod
    def get_file_handle(self, node, mode):
        # type: (File, str) -> io.IOBase
        pass

    @abc.abstractmethod
    def add_child_dir(self, parent_node, name):
        # type: (Dir_, unicode) -> Dir_
        pass

    @abc.abstractmethod
    def add_child_file(self, parent_node, name):
        # type: (Dir_, unicode) -> File_
        pass

    @abc.abstractmethod
    def remove_child(self, parent_node, name):
        # type: (Dir_, unicode) -> None
        pass
