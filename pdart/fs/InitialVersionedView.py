"""
Work on viewing an archive folder as a versioned filesystem.
"""
import io
from abc import *

from fs.base import FS
from fs.errors import DirectoryExpected, FileExpected, ResourceNotFound, \
    ResourceReadOnly
from fs.info import Info
from fs.mode import check_writable
from fs.osfs import OSFS
from fs.path import abspath, basename, iteratepath, join, normpath
from typing import TYPE_CHECKING

from pdart.fs.ReadOnlyView import ReadOnlyView
from pdart.fs.SubdirVersions import strSubdirVersions
from pdart.fs.VersionedFS import ROOT, SUBDIR_VERSIONS_FILENAME, \
    scan_vfs_dir

if TYPE_CHECKING:
    from typing import Any, AnyStr, List, Tuple


def _make_raw_dir_info(name):
    # type: (unicode) -> Dict
    return {u'basic': {u'name': name, u'is_dir': True}}


def _is_root(path):
    # type: (unicode) -> unicode
    return abspath(normpath(path)) == ROOT


def _is_version_part(part):
    # type: (unicode) -> bool
    return part.startswith(u'v$')


_ALL_PATS = [u'*']
# type: List[unicode]

VERSION_ONE = u'v$1'
# type: unicode

_VISIT_DIR_PAT = u'visit_*'
# type: unicode

FILE_EXCLUSION_PATS = [u'.DS_Store', u'*.db']


# type: List[unicode]


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


# The decision-making for paths is complex and trying to analyse the
# path and handle the different versions in getinfo(), listdir(), and
# openbin() just gets too complex and error-prone.  We use a standard
# object-oriented technique and make objects for the various path
# types, and have each implement their own version of these methods.

# Some of the paths are synthetic: that is, they don't match up with
# paths in the underlying filesystem, while some are just renames of
# existing paths.

# We have ten kinds of paths:
#     * the root path '/'
#     * the bundle path '/hst_14334'
#     * the collection path '/hst_14334/data_wfc3_raw'
#     * the product path '/hst_14334/data_wfc3_raw/icwy01hdq_raw'
#     * bundle version paths '/hst_14334/v$1'
#     * collection version paths '/hst_14334/data_wfc3_raw/v$1'
#     * product version paths '/hst_14334/data_wfc3_raw/icwy01hdq_raw/v$1'
#     * bundle version filepaths '/hst_14334/v$1/bundle.xml'
#     * collection version filepaths
#           '/hst_14334/data_wfc3_raw/v$1/collection.xml'
#     * product version filepaths
#           '/hst_14334/data_wfc3_raw/icwy01hdq_raw/v$1/icwy01hdq_raw.fits'


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
    A path that's a file.
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
        self._bundle = bundle

    def listdir(self):
        # type: () -> List[unicode]
        return [self._bundle]

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(u''))


class _FSBundlePath(_FSDirPath):
    """
    A path '/bundle'
    """

    def __init__(self, fs, path, bundle):
        # type: (OSFS, unicode, unicode) -> None
        _FSDirPath.__init__(self, fs, path)
        assert path == join(ROOT, bundle), '%s != %s' % (path, bundle)
        self._bundle = bundle

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(self._bundle))

    def listdir(self):
        # type: () -> List[unicode]
        dir_list = [info.name
                    for info in self._legacy_fs.scandir(ROOT)
                    if info.is_dir]
        dir_list.append(VERSION_ONE)
        return dir_list


class _FSCollectionPath(_FSDirPath):
    """
    A path '/bundle/collection'
    """

    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionedView) -> None
        _FSDirPath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]
        b, c = iteratepath(self._original_path)
        dir_list = list(self._big_fs._get_collection_fits_products(
            join(ROOT, c)))
        dir_list.append(VERSION_ONE)
        return dir_list


class _FSProductPath(_FSDirPath):
    """
    A path '/bundle/collection/product'
    """

    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        _FSDirPath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]
        return [VERSION_ONE]


class _FSBundleVersionDirPath(_FSDirPath):
    """
    A path '/bundle/v$1'
    """

    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        _FSDirPath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]

        # TODO: What if you've got a non-collection dir on the
        # top-level?  Conceptually, it belongs to the bundle.  How do
        # I distinguish?
        infos = self._legacy_fs.filterdir(
            ROOT,
            files=None,
            dirs=None,
            exclude_dirs=_ALL_PATS,
            exclude_files=FILE_EXCLUSION_PATS)
        return [info.name
                for info in infos
                if info.is_file] + [SUBDIR_VERSIONS_FILENAME]


class _FSCollectionVersionDirPath(_FSDirPath):
    """
    A path '/bundle/collection/v$1'
    """

    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        _FSDirPath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]
        b, c, v = iteratepath(self._original_path)
        infos = self._legacy_fs.filterdir(
            join(ROOT, c),
            files=None,
            dirs=None,
            exclude_dirs=_ALL_PATS,
            exclude_files=FILE_EXCLUSION_PATS)
        return [info.name
                for info in infos
                if info.is_file] + [SUBDIR_VERSIONS_FILENAME]


class _FSProductVersionDirPath(_FSDirPath):
    """
    A path '/bundle/collection/product/v$1'
    """

    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionedView) -> None
        _FSPath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, p, v = iteratepath(self._original_path)
        return Info(_make_raw_dir_info(v))

    def listdir(self):
        # type: () -> List[unicode]
        b, c, p, v = iteratepath(self._original_path)
        c_dir = join(ROOT, c)
        coll_prod_files = self._big_fs._get_collection_product_files(c_dir, p)
        return coll_prod_files + [SUBDIR_VERSIONS_FILENAME]


class _FSBundleVersionedFilePath(_FSFilePath):
    """
    A path '/bundle/v$1/file'
    """

    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        _FSFilePath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, v, f = iteratepath(self._original_path)
        return self._legacy_fs.getinfo(join(ROOT, f), namespaces)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, v, f = iteratepath(self._original_path)
        return self._legacy_fs.openbin(join(ROOT, f),
                                       mode,
                                       buffering,
                                       **options)


class _FSCollectionVersionedFilePath(_FSFilePath):
    """
    A path '/bundle/collection/v$1/file'
    """

    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionedView) -> None
        _FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, v, f = iteratepath(self._original_path)
        return self._legacy_fs.getinfo(join(ROOT, c, f))

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, c, v, f = iteratepath(self._original_path)
        return self._legacy_fs.openbin(join(ROOT, c, f),
                                       mode, buffering, **options)


class _FSProductVersionedFilePath(_FSFilePath):
    """
    A path '/bundle/collection/product/v$1/file'
    """

    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionedView) -> None
        _FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, p, v, f = iteratepath(self._original_path)
        fp = self._big_fs._get_product_filepath(join(ROOT, c), f)
        return self._legacy_fs.getinfo(fp)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, c, p, v, f = iteratepath(self._original_path)
        fp = self._big_fs._get_product_filepath(join(ROOT, c), f)
        return self._legacy_fs.openbin(fp, mode, buffering, **options)


class _FSSubdirVersionsFile(_FSFilePath):
    def __init__(self, fs, path, big_fs):
        _FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        return Info({u'basic': {u'name': SUBDIR_VERSIONS_FILENAME,
                                u'is_dir': False}})

    def openbin(self, mode, buffering, **options):
        ABOVE_VERSIONS = join(self._original_path, '..', '..')

        (ordinary_file_infos,
         ordinary_dir_infos,
         subdir_versions_file_infos,
         version_dir_infos) = scan_vfs_dir(self._big_fs, ABOVE_VERSIONS)

        d = dict([(info.name, u"1") for info in ordinary_dir_infos])
        return io.BytesIO(strSubdirVersions(d).encode('ascii'))


class InitialVersionedView(ReadOnlyView):
    """
    Wraps a filesystem in the unversioned legacy directory layout to
    look like a versioned filesystem with all PDS elements at version
    1.
    """

    def __init__(self, bundle_id, legacy_fs):
        # type: (unicode, FS) -> None
        self._bundle = bundle_id
        self._legacy_fs = legacy_fs
        # self._bundle = basename(bundle_dir)
        # self._legacy_fs = OSFS(bundle_dir)
        ReadOnlyView.__init__(self, self._legacy_fs)

    def _make_fs_path(self, path):
        # type: (unicode) -> _FSPath
        path = abspath(normpath(path))
        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            # synthetic root path
            return _FSRootPath(self._legacy_fs, ROOT, self._bundle)
        else:
            if not (parts[0] == self._bundle):
                # only files in the bundle exist in the filesystem
                raise ResourceNotFound(path)
            elif parts[-1] == SUBDIR_VERSIONS_FILENAME:
                return _FSSubdirVersionsFile(self._legacy_fs, path, self)
            if l == 1:
                # /bundle
                return _FSBundlePath(self._legacy_fs, path, self._bundle)
            elif l == 2:
                if _is_version_part(parts[1]):
                    # /bundle/version
                    return _FSBundleVersionDirPath(self._legacy_fs, path)
                else:
                    # /bundle/collection
                    return _FSCollectionPath(self._legacy_fs, path, self)
            elif l == 3:
                if _is_version_part(parts[1]):
                    # /bundle/version/file
                    return _FSBundleVersionedFilePath(self._legacy_fs, path)
                elif _is_version_part(parts[2]):
                    # /bundle/collection/version
                    return _FSCollectionVersionDirPath(self._legacy_fs, path)
                else:
                    # /bundle/collection/product
                    return _FSProductPath(self._legacy_fs, path)
            elif l == 4:
                if _is_version_part(parts[2]):
                    # /bundle/collection/version/file
                    return _FSCollectionVersionedFilePath(self._legacy_fs,
                                                          path,
                                                          self)
                else:
                    # /bundle/collection/product/version
                    assert _is_version_part(parts[3]), path
                    return _FSProductVersionDirPath(self._legacy_fs,
                                                    path,
                                                    self)
            elif l == 5:
                assert _is_version_part(parts[3])
                return _FSProductVersionedFilePath(self._legacy_fs, path, self)
            else:
                assert False, path

    def _get_visit_info(self, collection_path):
        return self._legacy_fs.filterdir(
            collection_path,
            files=None, dirs=[_VISIT_DIR_PAT],
            exclude_dirs=None, exclude_files=_ALL_PATS)

    def _get_collections(self):
        return [info.name
                for info
                in self._legacy_fs.filterdir(ROOT,
                                             files=None,
                                             dirs=['data_*'],
                                             exclude_dirs=None,
                                             exclude_files=_ALL_PATS)]

    def _get_collection_files(self, collection_path):
        # type: (unicode) -> List[Tuple[unicode, unicode]]
        return [(visit_info.name, file)
                for visit_info in self._get_visit_info(collection_path)
                for file in self._legacy_fs.listdir(join(collection_path,
                                                         visit_info.name))]

    def _get_collection_fits_products(self, collection_path):
        return [file[:-5]
                for (_, file) in self._get_collection_files(collection_path)
                if file.endswith(u'.fits')]

    def _get_collection_product_files(self, collection_path, product):
        return [file
                for (_, file) in self._get_collection_files(collection_path)
                if file.startswith(product)]

    def _get_product_filepath(self, collection_path, product_file):
        for (visit, file) in self._get_collection_files(collection_path):
            if file == product_file:
                return join(collection_path, visit, product_file)
        assert False, \
            '_get_product_filepath(%s, %s) not found' % \
            (collection_path, product_file)

    def _get_files(self, path):
        return [info.name for info in self._legacy_fs.filterdir(
            path,
            files=None,
            dirs=None,
            exclude_dirs=_ALL_PATS,
            exclude_files=FILE_EXCLUSION_PATS)]

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
