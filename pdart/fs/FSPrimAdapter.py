import fs.mode
import fs.path
from fs.base import FS
from fs.info import Info
from typing import TYPE_CHECKING

from pdart.fs.FSPrimitives import FSPrimitives

if TYPE_CHECKING:
    from typing import Tuple
    from pdart.fs.FSPrimitives import Node


class FSPrimAdapter(FS):
    def __init__(self, fs_prims):
        # type: (FSPrimitives) -> None
        FS.__init__(self)
        self.prims = fs_prims

    def getinfo(self, path, namespaces=None):
        # type: (unicode, List[str]) -> Info
        self.check()
        node = self._resolve_path_to_node(path)
        assert node, path
        info = {}
        info['basic'] = {
                'is_dir': self.prims.is_dir_prim(node),
                'name': fs.path.basename(node.name)
            }
        # info['details'] = {}
        return Info(info)

#    def getsyspath(self, path):
#        self.check()
#        prims = self.prims
#        node = self._resolve_path_to_node(path)
#        if node:
#            if prims.is_file():
#                return node.name
#            else:
#                return None

    def listdir(self, path):
        self.check()
        prims = self.prims
        node = self._resolve_path_to_node(path)
        if prims.is_file_prim(node):
            raise fs.errors.DirectoryExpected(path)
        else:
            return list(prims.get_dir_children(node))

    def makedir(self, path, permissions=None, recreate=False):
        # TODO What if it exists?
        self.check()
        parts = fs.path.iteratepath(path)
        if parts:
            parent_dir_node, name = self._resolve_path_to_parent_and_name(path)
            child_dir = self.prims.add_child_dir(parent_dir_node, name)
            return fs.subfs.SubFS(self, child_dir.name)
        else:
            if recreate:
                return fs.subfs.SubFS(self, self.prims.root_node().name)
            else:
                raise fs.errors.DirectoryExists(path)

    def openbin(self, path, mode="r", buffering=-1, **options):
        self.check()
        m = fs.mode.Mode(mode)
        m.validate_bin()
        prims = self.prims
        parent_dir_node, name = self._resolve_path_to_parent_and_name(path)

        if 't' in mode:
            raise ValueError('openbin() called with text mode %s', mode)
        exists = name in prims.get_children(parent_dir_node)
        if exists:
            file = prims.get_dir_child(parent_dir_node, name)
        elif m.create:
            file = prims.add_child_file(parent_dir_node, name)
        else:
            raise fs.errors.ResourceNotFound(path)
        return prims.get_handle(file, mode)

    def remove(self, path):
        self.check()
        prims = self.prims
        parent_dir_node, name = self._resolve_path_to_parent_and_name(path)
        try:
            dir = prims.get_dir_child(parent_dir_node, name)
        except KeyError:
            raise fs.errors.ResourceNotFound(path)
        if prims.is_file_prim(dir):
            prims.remove_child(parent_dir_node, name)
        else:
            raise fs.errors.FileExpected(path)

    def removedir(self, path):
        self.check()
        prims = self.prims
        try:
            parent_dir_node, name = self._resolve_path_to_parent_and_name(path)
        except IndexError:
            raise fs.errors.RemoveRootError(path)
        try:
            dir = prims.get_dir_child(parent_dir_node, name)
        except KeyError:
            raise fs.errors.ResourceNotFound(path)
        if prims.is_dir_prim(dir):
            if prims.get_dir_children(dir):
                raise fs.errors.DirectoryNotEmpty(path)
            else:
                prims.remove_child(parent_dir_node, name)
        else:
            raise fs.errors.DirectoryExpected(path)

    def setinfo(self, path, info):
        self.check()
        # Check for errors.
        self._resolve_path_to_node(path)
        # This is a no-op because the only info we use is 'basic' and
        # that's read-only.
        return None

    def _resolve_path_to_node(self, path):
        # type: (unicode) -> Node
        prims = self.prims
        node = prims.root_node()
        try:
            for nm in fs.path.iteratepath(path):
                node = prims.get_dir_child(node, nm)
            return node
        except KeyError:
            raise fs.errors.ResourceNotFound(path)

    def _resolve_path_to_parent_and_name(self, path):
        # type: (unicode) -> Tuple[Node, unicode]
        prims = self.prims
        node = prims.root_node()
        parts = fs.path.iteratepath(path)
        try:
            for nm in parts[:-1]:
                node = prims.get_dir_child(node, nm)
        except KeyError:
            raise fs.errors.ResourceNotFound(path)
        return (node, parts[-1])

    def open(self, path, mode='r', buffering=-1,
             encoding=None, errors=None, newline='', **options):
        # The FS implementation seems to be wrong (?!).  I'm replacing
        # it with something modelled off the OSFS implementation.
        bin_mode = mode.replace('t', '')
        return self.openbin(path, mode=bin_mode, buffering=buffering)
