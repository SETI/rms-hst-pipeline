"""
Work on viewing an archive folder as a versioned filesystem.
"""
from abc import *
import io
import pickle
import StringIO
import sys
import traceback
from typing import TYPE_CHECKING

from fs.base import FS
from fs.copy import copy_fs
from fs.errors import DirectoryExpected, FileExpected, InvalidPath, \
    ResourceNotFound, ResourceReadOnly
from fs.error_tools import unwrap_errors
from fs.info import Info
from fs.mode import check_writable
from fs.osfs import OSFS
from fs.path import abspath, basename, iteratepath, join, normpath
from fs.tempfs import TempFS

from pdart.fs.ReadOnlyView import ReadOnlyView
from pdart.fs.SubdirVersions import strSubdirVersions

if TYPE_CHECKING:
    from typing import Any, AnyStr, Tuple


_ROOT = u'/'
# type:unicode

_SUBDIR_VERSIONS_FILENAME = u'subdir$versions.txt'
# type: unicode


def _make_raw_dir_info(name):
    # type: (unicode) -> Dict
    return {u'basic': {u'name': name, u'is_dir': True}}


def _is_root(path):
    # type: (unicode) -> unicode
    return abspath(normpath(path)) == _ROOT


def _is_version_part(part):
    # type: (unicode) -> bool
    return part.startswith(u'v$')

_VERSION_ONE = u'v$1'
# type: unicode

_VISIT_DIR_PAT = u'visit_*'
# type: unicode

_VERSION_DICT = u'version$dict'
# type: unicode

ALL_PAT = u'*'
# type: unicode

FILE_EXCLUSION_PATS = [u'.DS_Store', '*.db']
# type: List[unicode]

_RAW_VERSION_DICT_INFO = {u'basic': {u'name': _VERSION_DICT, u'is_dir': False}}
# type: Dict


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
        assert path == _ROOT
        _FSDirPath.__init__(self, fs, _ROOT)
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
        assert path == join(_ROOT, bundle), '%s /= %s' % (path, bundle)
        self._bundle = bundle

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(self._bundle))

    def listdir(self):
        # type: () -> List[unicode]
        dir_list = [info.name
                    for info in self._legacy_fs.scandir(_ROOT)
                    if info.is_dir]
        dir_list.append(_VERSION_ONE)
        return dir_list


class _FSCollectionPath(_FSDirPath):
    """
    A path '/bundle/collection'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionView) -> None
        _FSDirPath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]
        b, c = iteratepath(self._original_path)
        dir_list = list(self._big_fs._get_collection_fits_products(
                join(_ROOT, c)))
        dir_list.append(_VERSION_ONE)
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
        return [_VERSION_ONE]


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
            _ROOT,
            files=None,
            dirs=None,
            exclude_dirs=[ALL_PAT],
            exclude_files=FILE_EXCLUSION_PATS)
        return [info.name
                for info in infos
                if info.is_file] + [_SUBDIR_VERSIONS_FILENAME]


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
            join(_ROOT, c),
            files=None,
            dirs=None,
            exclude_dirs=[ALL_PAT],
            exclude_files=FILE_EXCLUSION_PATS)
        return [info.name
                for info in infos
                if info.is_file] + [_SUBDIR_VERSIONS_FILENAME]


class _FSProductVersionDirPath(_FSDirPath):
    """
    A path '/bundle/collection/product/v$1'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionView) -> None
        _FSPath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, p, v = iteratepath(self._original_path)
        return Info(_make_raw_dir_info(v))

    def listdir(self):
        # type: () -> List[unicode]
        b, c, p, v = iteratepath(self._original_path)
        c_dir = join(_ROOT, c)
        return self._big_fs._get_collection_product_files(c_dir, p) + \
            [_SUBDIR_VERSIONS_FILENAME]


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
        return self._legacy_fs.getinfo(join(_ROOT, f), namespaces)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, v, f = iteratepath(self._original_path)
        return self._legacy_fs.openbin(join(_ROOT, f),
                                       mode,
                                       buffering,
                                       **options)


class _FSCollectionVersionedFilePath(_FSFilePath):
    """
    A path '/bundle/collection/v$1/file'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionView) -> None
        _FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, v, f = iteratepath(self._original_path)
        return self._legacy_fs.getinfo(join(_ROOT, c, f))

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, c, v, f = iteratepath(self._original_path)
        return self._legacy_fs.openbin(join(_ROOT, c, f),
                                       mode, buffering, **options)


class _FSProductVersionedFilePath(_FSFilePath):
    """
    A path '/bundle/collection/product/v$1/file'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, InitialVersionView) -> None
        _FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, p, v, f = iteratepath(self._original_path)
        fp = self._big_fs._get_product_filepath(join(_ROOT, c), f)
        return self._legacy_fs.getinfo(fp)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, c, p, v, f = iteratepath(self._original_path)
        fp = self._big_fs._get_product_filepath(join(_ROOT, c), f)
        return self._legacy_fs.openbin(fp, mode, buffering, **options)


class _FSSubdirVersionsFile(_FSFilePath):
    def __init__(self, fs, path, big_fs):
        _FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        return Info({u'basic': {u'name': _SUBDIR_VERSIONS_FILENAME,
                                u'is_dir': False}})

    def openbin(self, mode, buffering, **options):
        ABOVE_VERSIONS = join(self._original_path, '..', '..')
        print '>>>>', ABOVE_VERSIONS
        d = dict([(info.name, "1")
                  for info in self._big_fs.filterdir(ABOVE_VERSIONS,
                                                     None, None,
                                                     [u'v$*'], [u'*'])])
        return StringIO.StringIO(strSubdirVersions(d))


class InitialVersionView(ReadOnlyView):
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

    def _delegate_file_path(self, path):
        # type: (unicode) -> Tuple[FS, unicode]
        res = None, None
        # type: Tuple[FS, unicode]
        if not _is_root(path):
            parts = iteratepath(path)
            if parts[0] == self._bundle:
                if parts[1:]:
                    res = self._legacy_fs, join(*parts[1:])
                else:
                    res = self._legacy_fs, _ROOT
            else:
                raise ResourceNotFound(path)
        return res

    def _make_fs_path(self, path):
        # type: (unicode) -> _FSPath
        path = abspath(normpath(path))
        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            # synthetic root path
            return _FSRootPath(self._legacy_fs, _ROOT, self._bundle)
        else:
            if not (parts[0] == self._bundle):
                # only files in the bundle exist in the filesystem
                raise ResourceNotFound(path)
            elif parts[-1] == _SUBDIR_VERSIONS_FILENAME:
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
            exclude_dirs=None, exclude_files=[ALL_PAT])

    def _get_collections(self):
        return [info.name
                for info
                in self._legacy_fs.filterdir(_ROOT,
                                             files=None,
                                             dirs=['data_*'],
                                             exclude_dirs=None,
                                             exclude_files=[ALL_PAT])]

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
                files=None, dirs=None,
                exclude_dirs=[ALL_PAT], exclude_files=FILE_EXCLUSION_PATS)]

    def getinfo(self, path, namespaces=None):
        self.check()

        fs_path = self._make_fs_path(path)
        if fs_path:
            return fs_path.getinfo(namespaces)

        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            # path = /; purely synthetic
            raw_info = _make_raw_dir_info(u'')
        else:
            if not (parts[0] == self._bundle):
                raise ResourceNotFound(path)
            if l == 1:
                # path = /b; orig path = /
                raw_info = _make_raw_dir_info(self._bundle)
            elif l == 2:
                _b, c = parts
                if _is_version_part(c):
                    # path = b/$n
                    raw_info = _make_raw_dir_info(c)
                else:
                    # path = b/c; orig path = c
                    with unwrap_errors(path):
                        raw_info = self._legacy_fs.getinfo(c).raw
            elif l == 3:
                _b, c, p = parts
                if _is_version_part(c):
                    # path = b/$n/file; orig path = /file
                    if p == _VERSION_DICT:
                        raw_info = _RAW_VERSION_DICT_INFO
                    else:
                        with unwrap_errors(path):
                            raw_info = self._legacy_fs.getinfo(p).raw
                elif _is_version_part(p):
                    # path = b/c/$n; orig path = /c
                    raw_info = _make_raw_dir_info(p)
                else:
                    # path = b/c/p; synthetic
                    raw_info = _make_raw_dir_info(p)
            elif l == 4:
                _b, c, p, f = parts
                if _is_version_part(p):
                    # path = b/c/$n/file; orig path /c/file
                    if f == _VERSION_DICT:
                        raw_info = _RAW_VERSION_DICT_INFO
                    else:
                        with unwrap_errors(path):
                            raw_info = self._legacy_fs.getinfo(join(c, f)).raw
                elif _is_version_part(f):
                    # path = b/c/p/$n; synthetic
                    raw_info = _make_raw_dir_info(f)
                else:
                    assert False, 'getinfo(%s) has %d parts' % (path, l)
            elif l == 5:
                _b, c, p, v, f = parts
                # path = b/c/p/$n/f; orig path /c/visit_?/f
                assert _is_version_part(v), \
                    'getinfo(%s) has %d parts' % (path, l)
                with unwrap_errors(path):
                    filepath = self._get_product_filepath(c, f)
                    with unwrap_errors(path):
                        raw_info = self._legacy_fs.getinfo(filepath).raw
            else:
                assert False, 'getinfo(%s) has %d parts' % (path, l)
        return Info(raw_info)

    def openbin(self, path, mode="r", buffering=-1, **options):
        # type: (unicode, AnyStr, int, **Any) -> Any
        self.check()
        if check_writable(mode):
            raise ResourceReadOnly(path)

        fs_path = self._make_fs_path(path)
        if fs_path:
            return fs_path.openbin(mode, buffering, **options)

        print '#### openbin(%r, %r)' % (path, mode)
        if basename(path) == _VERSION_DICT:
            # is it the bundle or the collection?
            parts = iteratepath(path)
            if len(parts) == 3:
                b, vOne, vDict = parts
                if b == self._bundle and \
                        vOne == _VERSION_ONE and vDict == _VERSION_DICT:
                    d = dict((coll, _VERSION_ONE)
                             for coll in self._get_collections())
                    bytes = pickle.dumps(d)
                    return io.BytesIO(bytes)
                else:
                    # TODO b/c/v$1 should return ExpectedFile instead
                    raise ResourceNotFound(path)
            elif len(parts) == 4:
                b, c, vOne, vDict = parts
                if b == self._bundle and \
                        vOne == _VERSION_ONE and vDict == _VERSION_DICT:
                    d = dict((prod, _VERSION_ONE)
                             for prod
                             in self._get_collection_fits_products(c))
                    bytes = pickle.dumps(d)
                    return io.BytesIO(bytes)
                else:
                    raise ResourceNotFound(path)
            else:
                raise ResourceNotFound(path)
        _fs, _path = self._delegate_file_path(path)
        if _fs:
            with unwrap_errors(path):
                return _fs.openbin(_path, mode=mode,
                                   buffering=buffering, **options)
        else:
            assert False, 'InitialVersionView.openbin()'

    def listdir(self, path):
        self.check()

        fs_path = self._make_fs_path(path)
        if fs_path:
            return fs_path.listdir()

        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            dir_list = [self._bundle]
        else:
            if not (parts[0] == self._bundle):
                raise ResourceNotFound(path)
            if l == 1:
                # path = b
                dir_list = [info.name
                            for info in self._legacy_fs.scandir(_ROOT)
                            if info.is_dir]
                dir_list.append(_VERSION_ONE)
            elif l == 2:
                b, c = parts
                if _is_version_part(c):
                    # path = b/$n
                    dir_list = self._get_files(_ROOT)
                    dir_list.append(_VERSION_DICT)
                else:
                    # path = b/c
                    dir_list = list(self._get_collection_fits_products(c))
                    dir_list.append(_VERSION_ONE)
            elif l == 3:
                b, c, p = parts
                if _is_version_part(c):
                    # path = b/$n/p; orig path = /p
                    if self._legacy_fs.exists(p):
                        raise DirectoryExpected(path)
                    else:
                        raise ResourceNotFound(path)
                elif _is_version_part(p):
                    # path = b/c/$n; orig path = /c
                    dir_list = self._get_files(c)
                    dir_list.append(_VERSION_DICT)
                else:
                    # path = b/c/p
                    dir_list = [_VERSION_ONE]
            elif l == 4:
                b, c, p, f = parts
                if _is_version_part(p):
                    # path = b/c/$n/f; orig path /c/f
                    if self._legacy_fs.exists(join(c, f)):
                        raise DirectoryExpected(path)
                    else:
                        raise ResourceNotFound(path)
                elif _is_version_part(f):
                    # path= b/c/p/$i; orig path /c/v*
                    dir_list = list(self._get_collection_product_files(c, p))
                else:
                    assert False, 'listdir(%s) has %d parts' % (path, l)
            elif l == 5:
                assert False, 'listdir(%s) has %d parts' % (path, l)
            else:
                assert False, 'listdir(%s) has %d parts' % (path, l)
        return dir_list

    def _get_versions_dict(self, path):
        bs = self.getbytes(join(path, _VERSION_DICT))
        return pickle.loads(bs)


def test_fs():
    fs = OSFS(u'/Users/spaceman/Desktop/Archive/hst_14334')
    return InitialVersionView(u'hst_14334', fs)


if __name__ == '__main__':
    fs0 = OSFS(u'/Users/spaceman/Desktop/Archive/hst_14334')
    fs1 = InitialVersionView('hst_14334', fs0)
    # import pudb; pu.db
    if False:
        print "fs1.listdir(u'/hst_14334') = %s" % \
            list(fs1.listdir(u'/hst_14334'))
        print "fs1.scandir(u'/hst_14334') = %s" % \
            list(fs1.scandir(u'/hst_14334'))
        print "fs1.filterdir(u'/hst_14334') = %s" % \
            list(fs1.filterdir(u'/hst_14334'))
        print fs1._get_collection_fits_products(u'data_wfc3_raw')
        print fs1._get_versions_dict(u'/hst_14334/v$1')
        print fs1._get_versions_dict(u'/hst_14334/data_wfc3_raw/v$1')
    print fs1
    fs1.tree()

    fs2 = TempFS(u'trashme')
    copy_fs(fs1, fs2)
