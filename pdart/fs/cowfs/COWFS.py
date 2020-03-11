import fs.copy
import fs.errors
from fs.info import Info
import fs.path
from fs.subfs import SubFS
import fs.wrap
from fs.base import FS
from fs.mode import Mode
from fs.tempfs import TempFS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Generator, List, Mapping, Optional, Text
    from fs.permissions import Permissions
    from fs.subfs import SubFS
    _INFO_DICT = Mapping[unicode, Mapping[unicode, object]]

NO_LAYER = 0  # exists in deletion layer
BASE_LAYER = 1  # exists in base (original, read-only) layer
ADD_LAYER = 2  # exists in addition layer
ROOT_LAYER = 3  # not really a layer but a special case

layer_names = ['NO_LAYER', 'BASE_LAYER', 'ADD_LAYER', 'ROOT_LAYER']

DEL = '__DELETED__'

def _deletions_invariant(filesys):
    # type: (FS) -> None
    """
    The only files the given filesystem contains all are named DEL.
    """
    names = {fs.path.basename(file) for file in filesys.walk.files()}
    assert not names or names == {DEL}


def paths(filesys):
    # type: (FS) -> Generator[unicode, None, None]
    """
    All the directory and file names in the filesystem.
    """
    for dir in filesys.walk.dirs():
        yield dir
    for file in filesys.walk.files():
        yield file


def del_path(path):
    # type: (unicode) -> unicode
    """
    Create the filepath for the deletion marker file.
    """
    return fs.path.combine(path, DEL)


class COWFS(FS):
    def __init__(self, base_fs, additions_fs=None, deletions_fs=None):
        # type: (FS, Optional[FS], Optional[FS]) -> None
        FS.__init__(self)
        if additions_fs:
            self.additions_fs = additions_fs
        else:
            self.additions_fs = TempFS()

        if deletions_fs:
            _deletions_invariant(deletions_fs)
            self.deletions_fs = deletions_fs
        else:
            self.deletions_fs = TempFS()

        self.original_base_fs = base_fs
        self.base_fs = fs.wrap.read_only(base_fs)

        self.invariant()

    @staticmethod
    def create_cowfs(base_fs, read_write_layer, recreate=False):
        # type: (FS, FS, bool) -> COWFS
        additions_fs = read_write_layer.makedir(u'/additions',
                                                recreate=recreate)
        deletions_fs = read_write_layer.makedir(u'/deletions',
                                                recreate=recreate)

        return COWFS(base_fs, additions_fs, deletions_fs)

    def __str__(self):
        return 'COWFS(%s, %s, %s)' % (self.original_base_fs,
                                      self.additions_fs,
                                      self.deletions_fs)

    def __repr__(self):
        return 'COWFS(%r, %r, %r)' % (self.original_base_fs,
                                      self.additions_fs,
                                      self.deletions_fs)

    ############################################################

    def invariant(self):
        # type: () -> bool
        assert self.additions_fs
        assert self.deletions_fs
        assert self.base_fs

        _deletions_invariant(self.deletions_fs)

        additions_paths = set(paths(self.additions_fs))
        deletions_paths = {fs.path.dirname(file)
                           for file
                           in self.deletions_fs.walk.files()}
        assert additions_paths <= deletions_paths, \
            'additions_paths %s is not a subset of deletions_path %s' % \
            (additions_paths, deletions_paths)

        return True

    def is_deletion(self, path):
        # type: (unicode) -> bool
        """
        Is the path marked in the deletions_fs"
        """
        return self.deletions_fs.exists(del_path(path))

    def mark_deletion(self, path):
        # type: (unicode) -> None
        """
        Mark the path in the deletions_fs.
        """
        self.deletions_fs.makedirs(path, None, True)
        self.deletions_fs.touch(del_path(path))

    def makedirs_mark_deletion(self, path, permissions=None, recreate=False):
        for p in fs.path.recursepath(path)[:-1]:
            self.additions_fs.makedirs(p,
                                       permissions=permissions,
                                       recreate=True)
            self.mark_deletion(p)
        self.additions_fs.makedir(path,
                                  permissions=permissions,
                                  recreate=recreate)
        self.mark_deletion(path)

    def layer(self, path):
        # type: (unicode) -> int
        """
        Get the layer on which the file lives, or ROOT_LAYER if it's the
        root path.
        """

        if path == u'/':
            return ROOT_LAYER
        if self.additions_fs.exists(path):
            return ADD_LAYER
        elif self.is_deletion(path):
            return NO_LAYER
        elif self.base_fs.exists(path):
            return BASE_LAYER
        else:
            return NO_LAYER

    def copy_up(self, path):
        # type: (unicode) -> None
        """
        Copy the file from the base_fs to additions_fs.
        """
        self.makedirs_mark_deletion(fs.path.dirname(path))
        self.mark_deletion(path)
        fs.copy.copy_file(self.base_fs,
                          path,
                          self.additions_fs,
                          path)

    def triple_tree(self):
        # type: () -> None
        print 'base_fs ------------------------------'
        self.base_fs.tree()
        print 'additions_fs ------------------------------'
        self.additions_fs.tree()
        print 'deletions_fs ------------------------------'
        self.deletions_fs.tree()


    ############################################################

    def getmeta(self, namespace='standard'):
        # type: (Text) -> Mapping[Text, object]
        return  self.base_fs.getmeta(namespace)

    def getinfo(self, path, namespaces=None):
        # type: (unicode, Optional[List[str]]) -> Info
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            return self.base_fs.getinfo(path, namespaces)
        elif layer == ADD_LAYER:
            return self.additions_fs.getinfo(path, namespaces)
        elif layer == ROOT_LAYER:
            # TODO implement this
            raw_info = {}
            if namespaces is None or 'basic' in namespaces:
                raw_info[u'basic'] = { u'name': u'', u'is_dir': True }
            return Info(raw_info)
        else:
            assert False, 'unknown layer %d' % (layer,)

    def getsyspath(self, path):
        # type: (unicode) -> unicode
        self.check()
        # self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.NoSysPath(path=path)
        elif layer == BASE_LAYER:
            return self.base_fs.getsyspath(path)
        elif layer == ADD_LAYER:
            return self.additions_fs.getsyspath(path)
        elif layer == ROOT_LAYER:
            raise fs.errors.NoSysPath(path=path)
        else:
            assert False, 'unknown layer %d' % (layer,)

    def listdir(self, path):
        # type: (unicode) -> List[unicode]
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            return self.base_fs.listdir(path)
        elif layer == ADD_LAYER:
            # Get the listing on the additions layer
            names = set(self.additions_fs.listdir(path))
            # Add in the listing on the base layer (if it exists)
            if self.base_fs.isdir(path):
                names |= set(self.base_fs.listdir(path))
            # Return the entries that actually exist
            return [name for name in list(names)
                    if self.layer(fs.path.join(path, name)) != NO_LAYER]
        elif layer == ROOT_LAYER:
            # Get the listing of the root on the additions layer and
            # the base layer.
            names = set(self.additions_fs.listdir(u'/'))
            names |= set(self.base_fs.listdir(u'/'))
            # Return the entries that actually exist.
            return [name for name in list(names)
                    if self.layer(name) != NO_LAYER]
        else:
            assert False, 'unknown layer %d' % (layer,)

    def makedir(self, path, permissions=None, recreate=False):
        # type: (unicode, Optional[Permissions], bool) -> SubFS
        self.check()
        self.validatepath(path)

        # Check if it *can* be created.

        # get a normalized parent_dir path.
        parent_dir = fs.path.dirname(fs.path.forcedir(path)[:-1])
        if not parent_dir:
            parent_dir = u'/'

        if not self.isdir(parent_dir):
            raise fs.errors.ResourceNotFound(path)

        layer = self.layer(path)
        if layer == NO_LAYER:
            self.makedirs_mark_deletion(path,
                                        permissions=permissions,
                                        recreate=recreate)
            return SubFS(self, path)
        elif layer in [BASE_LAYER, ADD_LAYER, ROOT_LAYER]:
            if recreate:
                return SubFS(self, path)
            else:
                # I think this is wrong.  What if it's a file?
                raise fs.errors.DirectoryExists(path)
        else:
            assert False, 'unknown layer %d' % (layer,)

    def openbin(self, path, mode='r', buffering=-1, **options):
        self.check()
        self.validatepath(path)

        parent_dir = fs.path.dirname(fs.path.forcedir(path)[:-1])
        if not parent_dir:
            parent_dir = u'/'

        if not self.isdir(parent_dir):
            raise fs.errors.ResourceNotFound(path)

        mode_obj = Mode(mode)
        layer = self.layer(path)
        if layer == NO_LAYER:
            if mode_obj.create:
                for p in fs.path.recursepath(path)[:-1]:
                    self.additions_fs.makedirs(p,
                                               recreate=True)
                    self.mark_deletion(p)
                self.mark_deletion(path)
                return self.additions_fs.openbin(path, mode,
                                                 buffering, **options)
            else:
                raise fs.errors.ResourceNotFound(path)
        elif layer == ADD_LAYER:
            self.mark_deletion(path)
            return self.additions_fs.openbin(path, mode, buffering, **options)
        elif layer == BASE_LAYER:
            if mode_obj.writing:
                self.copy_up(path)
                return self.additions_fs.openbin(path, mode,
                                                 buffering, **options)
            else:
                return self.base_fs.openbin(path, mode, buffering, **options)
        elif layer == ROOT_LAYER:
            raise fs.errors.FileExpected(path)
        else:
            assert False, 'unknown layer %d' % (layer,)

    def remove(self, path):
        # type: (unicode) -> None
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            if self.base_fs.isfile(path):
                self.mark_deletion(path)
            else:
                raise fs.errors.FileExpected(path)
        elif layer == ADD_LAYER:
            self.additions_fs.remove(path)
            self.mark_deletion(path)
        elif layer == ROOT_LAYER:
            raise fs.errors.FileExpected(path)
        else:
            assert False, 'unknown layer %d' % (layer,)

    def removedir(self, path):
        # type: (unicode) -> None
        self.check()
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            if self.base_fs.isdir(path):
                self.mark_deletion(path)
            else:
                raise fs.errors.FileExpected(path)
        elif layer == ADD_LAYER:
            if self.additions_fs.isdir(path):
                self.additions_fs.removedir(path)
                self.mark_deletion(path)
            else:
                raise fs.errors.DirectoryExpected(path)
        elif layer == ROOT_LAYER:
            raise fs.errors.RemoveRootError(path)
        else:
            assert False, 'unknown layer %d' % (layer,)

    def setinfo(self, path, info):
        # type: (unicode, _INFO_DICT) -> None
        self.check()
        self.validatepath(path)
        layer = self.layer(path)
        if layer == NO_LAYER:
            raise fs.errors.ResourceNotFound(path)
        elif layer == BASE_LAYER:
            self.copy_up(path)
            self.additions_fs.setinfo(path, info)
        elif layer == ADD_LAYER:
            self.additions_fs.setinfo(path, info)
        elif layer == ROOT_LAYER:
            pass
        else:
            assert False, 'unknown layer %d' % (layer,)

    ############################################################

    def makedirs(self, path, permissions=None, recreate=False):
        return FS.makedirs(self,
                           path,
                           permissions=permissions,
                           recreate=recreate)
