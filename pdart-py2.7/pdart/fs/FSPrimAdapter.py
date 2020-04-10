from fs.base import FS
from fs.enums import ResourceType
from fs.error_tools import convert_os_errors
from fs.info import Info
import fs.mode
import fs.path
import fs.subfs
import os
import stat
from typing import TYPE_CHECKING, cast

from pdart.fs.FSPrimitives import FSPrimitives

if TYPE_CHECKING:
    from typing import Any, Dict, Text, Tuple, Union
    from pdart.fs.FSPrimitives import Node_

_WINDOWS_PLATFORM = False


class FSPrimAdapter(FS):
    def __init__(self, fs_prims):
        # type: (FSPrimitives) -> None
        FS.__init__(self)
        self.prims = fs_prims

        _meta = self._meta = {
            "case_insensitive": os.path.normcase("Aa") != "aa",
            "network": False,
            "read_only": False,
            "supports_rename": False,
            "thread_safe": True,
            "unicode_paths": False,
            "virtual": False,
            "invalid_path_chars": "\0",
        }

    def getinfo(self, path, namespaces=None):
        # type: (unicode, Any) -> Info

        # The pyfilesystem2 documentation says namespaces should be a
        # list of strings, but the test-suite has a case expecting it
        # to succeed when it's a single string.  Geez.

        # I REALLY REALLY hate untyped languages.
        self.check()
        if not namespaces:
            namespaces = ["basic"]
        if type(namespaces) is not list:
            namespaces = [namespaces]
        node = self._resolve_path_to_node(path)
        assert node, path
        info = {}  # Dict[unicode, Dict[unicode, object]]
        info[u"basic"] = {
            u"is_dir": self.prims.is_dir(node),
            u"name": fs.path.basename(node.path),
        }
        if "details" in namespaces:
            sys_path = self.getsyspath(path)
            if sys_path:
                with convert_os_errors("getinfo", path):
                    _stat = os.stat(sys_path)
                info[u"details"] = self._make_details_from_stat(_stat)
            else:
                info[u"details"] = self._make_default_details(node)

        return Info(info)

    def listdir(self, path):
        self.check()
        prims = self.prims
        node = self._resolve_path_to_node(path)
        if prims.is_file(node):
            raise fs.errors.DirectoryExpected(path)
        else:
            return list(prims.get_dir_children(node))

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        parts = fs.path.iteratepath(fs.path.abspath(path))
        if not parts:  # we're looking at the root
            if recreate:
                return fs.subfs.SubFS(self, self.prims.root_node().path)
            else:
                raise fs.errors.DirectoryExists(path)
        else:
            prims = self.prims
            parent_dir_node, name = self._resolve_path_to_parent_and_name(path)
            try:
                child = prims.get_dir_child(parent_dir_node, name)
            except KeyError:
                # it doesn't exist
                child = prims.add_child_dir(parent_dir_node, name)
                return fs.subfs.SubFS(self, child.path)
            # it exists
            if prims.is_file(child):
                # TODO This is wrong, but the pyfilesystem test suite
                # asks for it...
                raise fs.errors.DirectoryExists(path)
            else:
                if recreate:
                    return fs.subfs.SubFS(self, child.path)
                else:
                    raise fs.errors.DirectoryExists(path)

    def openbin(self, path, mode="r", buffering=-1, **options):
        self.check()
        self.validatepath(path)
        if path == u"/":
            # TODO  Hackish special case.  Clean this up.
            raise fs.errors.FileExpected(path)
        m = fs.mode.Mode(mode)
        m.validate_bin()
        prims = self.prims
        parent_dir_node, name = self._resolve_path_to_parent_and_name(path)

        if "t" in mode:
            raise ValueError("openbin() called with text mode %s", mode)
        exists = prims.is_dir(parent_dir_node) and name in prims.get_children(
            parent_dir_node
        )
        if exists:
            if m.exclusive:
                raise fs.errors.FileExists(path)
            else:
                file = prims.get_dir_child(parent_dir_node, name)
        elif m.create:
            file = prims.add_child_file(parent_dir_node, name)
        else:
            raise fs.errors.ResourceNotFound(path)
        return prims.get_handle(file, m.to_platform())

    def remove(self, path):
        self.check()
        prims = self.prims
        parent_dir_node, name = self._resolve_path_to_parent_and_name(path)
        try:
            dir = prims.get_dir_child(parent_dir_node, name)
        except KeyError:
            raise fs.errors.ResourceNotFound(path)
        if prims.is_file(dir):
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
        if prims.is_dir(dir):
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
        if "details" in info:
            sys_path = self.getsyspath(path)
            if sys_path:
                details = info["details"]
                if "accessed" in details or "modified" in details:
                    _accessed = cast(int, details.get("accessed"))
                    _modified = cast(int, details.get("modified", _accessed))
                    accessed = int(_modified if _accessed is None else _accessed)
                    modified = int(_modified)
                    if accessed is not None or modified is not None:
                        with convert_os_errors("setinfo", path):
                            os.utime(sys_path, (accessed, modified))

        return None

    def _resolve_path_to_node(self, path):
        # type: (unicode) -> Node_
        prims = self.prims
        node = prims.root_node()
        try:
            for nm in fs.path.iteratepath(path):
                node = prims.get_dir_child(node, nm)
            return node
        except KeyError:
            raise fs.errors.ResourceNotFound(path)

    def _resolve_path_to_parent_and_name(self, path):
        # type: (unicode) -> Tuple[Node_, unicode]
        prims = self.prims
        node = prims.root_node()
        parts = fs.path.iteratepath(path)
        try:
            for nm in parts[:-1]:
                node = prims.get_dir_child(node, nm)
        except KeyError as e:
            print "e =", e
            print "nm =", repr(nm)
            print (
                "prims.get_dir_children(%r) = %r" % (node, prims.get_dir_children(node))
            )
            raise fs.errors.ResourceNotFound(path)
        return (node, parts[-1])

    @classmethod
    def _make_details_from_stat(cls, stat_result):
        # type: (Any) -> Dict[unicode, Any]
        """Make a *details* info dict from an `os.stat_result` object.
        """
        details = {
            "_write": ["accessed", "modified"],
            "accessed": stat_result.st_atime,
            "modified": stat_result.st_mtime,
            "size": stat_result.st_size,
            "type": int(cls._get_type_from_stat(stat_result)),
        }  # type: Dict[unicode, Any]

        # On other Unix systems (such as FreeBSD), the following
        # attributes may be available (but may be only filled out if
        # root tries to use them):
        details["created"] = getattr(stat_result, "st_birthtime", None)
        ctime_key = "created" if _WINDOWS_PLATFORM else "metadata_changed"
        details[ctime_key] = stat_result.st_ctime
        return details

    def _make_default_details(self, node):
        # type: (Node_) -> Dict[unicode, Any]
        """Make a default *details* info dict"""
        prims = self.prims
        if prims.is_dir(node):
            resource_type = ResourceType.directory
        elif prims.is_file(node):
            resource_type = ResourceType.file
        else:
            resource_type = ResourceType.unknown

        details = {
            "accessed": None,
            "created": None,
            "modified": None,
            "size": 0,
            "type": resource_type,
        }  # type: Dict[unicode, Any]
        return details

    @classmethod
    def _get_type_from_stat(cls, _stat):
        # type: (Any) -> ResourceType
        """Get the resource type from an `os.stat_result` object.
        """
        st_mode = _stat.st_mode
        st_type = stat.S_IFMT(st_mode)
        return cls.STAT_TO_RESOURCE_TYPE.get(st_type, ResourceType.unknown)

    STAT_TO_RESOURCE_TYPE = {
        stat.S_IFDIR: ResourceType.directory,
        stat.S_IFCHR: ResourceType.character,
        stat.S_IFBLK: ResourceType.block_special_file,
        stat.S_IFREG: ResourceType.file,
        stat.S_IFIFO: ResourceType.fifo,
        stat.S_IFLNK: ResourceType.symlink,
        stat.S_IFSOCK: ResourceType.socket,
    }
