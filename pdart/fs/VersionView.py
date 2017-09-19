from abc import *
from typing import TYPE_CHECKING

from fs.errors import DirectoryExpected, FileExpected, ResourceNotFound, \
    ResourceReadOnly
from fs.info import Info
from fs.mode import check_writable
from fs.path import abspath, basename, iteratepath, join, normpath

from pdart.fs.ReadOnlyView import ReadOnlyView
from pdart.fs.SubdirVersions import readSubdirVersions
from pdart.fs.VersionedFS import ROOT, SUBDIR_VERSIONS_FILENAME

if TYPE_CHECKING:
    from typing import Any, AnyStr, Tuple
    from fs.base import FS
    from fs.osfs import OSFS


def _make_raw_dir_info(name):
    # type: (unicode) -> Dict
    return {u'basic': {u'name': name, u'is_dir': True}}


class _FSPath(object):
    __metaclass__ = ABCMeta

    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        self._legacy_fs = fs
        self._original_path = path

    def __str__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self._legacy_fs,
                               self._original_path)

    @abstractmethod
    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        pass

    @abstractmethod
    def listdir(self):
        # type: () -> List[unicode]
        pass

    @abstractmethod
    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        pass


class _FSDirPath(_FSPath):
    """
    A path that's a directory.
    """
    def __init__(self, fs, path):
        _FSPath.__init__(self, fs, path)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        raise FileExpected(self._original_path)


class _FSFilePath(_FSPath):
    """
    A path that's a file
    """
    def __init__(self, fs, path):
        _FSPath.__init__(self, fs, path)

    def listdir(self):
        # type: () -> List[unicode]
        raise DirectoryExpected(self._original_path)


class _FSRootPath(_FSDirPath):
    """
    The root path '/'
    """
    def __init__(self, fs, path, bundle):
        # type: (OSFS, unicode, unicode) -> None
        assert path == ROOT
        _FSDirPath.__init__(self, fs, ROOT)
        self._bundle_id = bundle

    def listdir(self):
        # type: () -> List[unicode]
        return [self._bundle_id]

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(u''))


class _FSBundlePath(_FSDirPath):
    """
    A path '/bundle'
    """
    def __init__(self, fs, path, bundle, bundle_version):
        # type: (OSFS, unicode, unicode, unicode) -> None
        _FSDirPath.__init__(self, fs, path)
        assert path == join(ROOT, bundle), '%s != %s' % (path, bundle)
        self._bundle_id = bundle
        self._bundle_version = bundle_version

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(self._bundle_id))

    def listdir(self):
        # type: () -> List[unicode]
        legacy_dirpath = join(u'/',
                              self._bundle_id,
                              'v%s' % self._bundle_version)
        files = [info.name
                 for info in self._legacy_fs.scandir(legacy_dirpath)
                 if info.is_file and not info.name == SUBDIR_VERSIONS_FILENAME]
        versions = readSubdirVersions(self._legacy_fs, legacy_dirpath)
        dirs = versions.keys()
        # TODO maybe assert that they're disjoint
        return files + dirs


class _FSBundleFile(_FSFilePath):
    """
    A path '/bundle/file'
    """
    def __init__(self, fs, path, legacy_path):
        _FSFilePath.__init__(self, fs, path)
        self._legacy_path = legacy_path

    def getinfo(self, namespaces):
        return self._legacy_fs.getinfo(self._legacy_path)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        return self._legacy_fs.openbin(mode, buffering, **options)


class VersionView(ReadOnlyView):
    def __init__(self, bundle_lidvid, wrap_fs):
        # type: (unicode, FS) -> None
        bundle_lid, self._version_id = bundle_lidvid.split(u'::')[-2:]
        self._bundle_id = bundle_lid.split(u':')[-1]
        assert wrap_fs.exists(join(u'/', self._bundle_id,
                                   u'v%s' % self._version_id))
        self._legacy_fs = wrap_fs
        ReadOnlyView.__init__(self, self._legacy_fs)

    def _make_fs_path(self, path):
        # type: (unicode) -> _FSPath
        path = abspath(normpath(path))
        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            # synthetic root path
            return _FSRootPath(self._legacy_fs, ROOT, self._bundle_id)
        else:
            if not (parts[0] == self._bundle_id):
                # only files in the bundle exist in the filesystem
                raise ResourceNotFound(path)
            if l == 1:
                # /bundle
                return _FSBundlePath(self._legacy_fs,
                                     path,
                                     self._bundle_id,
                                     self._version_id)
            elif l == 2:
                # /bundle/file
                return _FSBundleFile(self._legacy_fs, path,
                                     join(u'/', self._bundle_id,
                                          u'v%s' % self._version_id,
                                          basename(path)))
            else:
                assert False, '_make_fs_path(%r) unimplemented' % path

    def getinfo(self, path, namespaces=None):
        self.check()
        fs_path = self._make_fs_path(path)
        return fs_path.getinfo(namespaces)

    def openbin(self, path, mode="r", buffering=-1, **options):
        # type: (unicode, AnyStr, int, **Any) -> Any
        self.check()
        if check_writable(mode):
            raise ResourceReadOnly(path)

        fs_path = self._make_fs_path(path)
        return fs_path.openbin(mode, buffering, **options)

    def listdir(self, path):
        self.check()

        fs_path = self._make_fs_path(path)
        return fs_path.listdir()
