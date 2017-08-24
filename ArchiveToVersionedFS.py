"""
Work on viewing an archive folder as a versioned filesystem.
"""
from abc import *
import io
import pickle
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

if TYPE_CHECKING:
    from typing import Any, AnyStr, Tuple


_ROOT = u'/'
# type:unicode


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


class FSPath(object):
    __metaclass__ = ABCMeta

    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        self._wrap_fs = fs
        self._original_path = path

    def __str__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self._wrap_fs,
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


class FSDirPath(FSPath):
    """
    A path that's a directory.
    """
    def __init__(self, fs, path):
        FSPath.__init__(self, fs, path)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        raise FileExpected(self._original_path)


class FSFilePath(FSPath):
    """
    A path that's a file.
    """
    def __init__(self, fs, path):
        FSPath.__init__(self, fs, path)

    def listdir(self):
        # type: () -> List[unicode]
        raise DirectoryExpected(self._original_path)


class FSRootPath(FSDirPath):
    """
    The root path '/'
    """
    def __init__(self, fs, path, bundle):
        # type: (OSFS, unicode, unicode) -> None
        assert path == _ROOT
        FSDirPath.__init__(self, fs, _ROOT)
        self._bundle = bundle

    def listdir(self):
        # type: () -> List[unicode]
        return [self._bundle]

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(u''))


class FSBundlePath(FSDirPath):
    """
    A path '/bundle'
    """
    def __init__(self, fs, path, bundle):
        # type: (OSFS, unicode, unicode) -> None
        FSDirPath.__init__(self, fs, path)
        assert path == join(_ROOT, bundle), '%s /= %s' % (path, bundle)
        self._bundle = bundle

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(self._bundle))

    def listdir(self):
        # type: () -> List[unicode]
        dir_list = [info.name
                    for info in self._wrap_fs.scandir(_ROOT)
                    if info.is_dir]
        dir_list.append(_VERSION_ONE)
        return dir_list


class FSCollectionPath(FSDirPath):
    """
    A path '/bundle/collection'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, ArchiveToVersionedFS) -> None
        FSDirPath.__init__(self, fs, path)
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


class FSProductPath(FSDirPath):
    """
    A path '/bundle/collection/product'
    """
    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        FSDirPath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]
        return [_VERSION_ONE]


class FSBundleVersionDirPath(FSDirPath):
    """
    A path '/bundle/v$1'
    """
    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        FSDirPath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]

        # TODO: What if you've got a non-collection dir on the
        # top-level?  Conceptually, it belongs to the bundle.  How do
        # I distinguish?
        infos = self._wrap_fs.filterdir(
            _ROOT,
            files=None,
            dirs=None,
            exclude_dirs=[ALL_PAT],
            exclude_files=FILE_EXCLUSION_PATS)
        return [info.name
                for info in infos
                if info.is_file]


class FSCollectionVersionDirPath(FSDirPath):
    """
    A path '/bundle/collection/v$1'
    """
    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        FSDirPath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        return Info(_make_raw_dir_info(basename(self._original_path)))

    def listdir(self):
        # type: () -> List[unicode]
        b, c, v = iteratepath(self._original_path)
        infos = self._wrap_fs.filterdir(
            join(_ROOT, c),
            files=None,
            dirs=None,
            exclude_dirs=[ALL_PAT],
            exclude_files=FILE_EXCLUSION_PATS)
        return [info.name
                for info in infos
                if info.is_file]


class FSProductVersionDirPath(FSDirPath):
    """
    A path '/bundle/collection/product/v$1'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, ArchiveToVersionedFS) -> None
        FSPath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, p, v = iteratepath(self._original_path)
        return Info(_make_raw_dir_info(v))

    def listdir(self):
        # type: () -> List[unicode]
        b, c, p, v = iteratepath(self._original_path)
        return self._big_fs._get_collection_product_files(join(_ROOT, c), p)


class FSBundleVersionedFilePath(FSFilePath):
    """
    A path '/bundle/v$1/file'
    """
    def __init__(self, fs, path):
        # type: (OSFS, unicode) -> None
        FSFilePath.__init__(self, fs, path)

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, v, f = iteratepath(self._original_path)
        return self._wrap_fs.getinfo(join(_ROOT, f), namespaces)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, v, f = iteratepath(self._original_path)
        return self._wrap_fs.openbin(join(_ROOT, f),
                                     mode,
                                     buffering,
                                     **options)


class FSCollectionVersionedFilePath(FSFilePath):
    """
    A path '/bundle/collection/v$1/file'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, ArchiveToVersionedFS) -> None
        FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, v, f = iteratepath(self._original_path)
        return self._wrap_fs.getinfo(join(_ROOT, c, f))

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, c, v, f = iteratepath(self._original_path)
        return self._wrap_fs.openbin(join(_ROOT, c, f),
                                     mode, buffering, **options)


class FSProductVersionedFilePath(FSFilePath):
    """
    A path '/bundle/collection/product/v$1/file'
    """
    def __init__(self, fs, path, big_fs):
        # type: (OSFS, unicode, ArchiveToVersionedFS) -> None
        FSFilePath.__init__(self, fs, path)
        self._big_fs = big_fs

    def getinfo(self, namespaces):
        # type: (Tuple) -> Info
        b, c, p, v, f = iteratepath(self._original_path)
        fp = self._big_fs._get_product_filepath(join(_ROOT, c), f)
        return self._wrap_fs.getinfo(fp)

    def openbin(self, mode, buffering, **options):
        # type: (AnyStr, int, **Any) -> Any
        b, c, p, v, f = iteratepath(self._original_path)
        fp = self._big_fs._get_product_filepath(join(_ROOT, c), f)
        return self._wrap_fs.openbin(fp, mode, buffering, **options)


class ArchiveToVersionedFS(ReadOnlyView):
    def __init__(self, bundle_dir):
        # type: (unicode) -> None
        self._bundle_dir = bundle_dir
        self._bundle = basename(bundle_dir)
        self._wrap_fs = OSFS(bundle_dir)
        ReadOnlyView.__init__(self, self._wrap_fs)

    def _delegate_file_path(self, path):
        # type: (unicode) -> Tuple[FS, unicode]
        res = None, None
        # type: Tuple[FS, unicode]
        if not _is_root(path):
            parts = iteratepath(path)
            if parts[0] == self._bundle:
                if parts[1:]:
                    res = self._wrap_fs, join(*parts[1:])
                else:
                    res = self._wrap_fs, _ROOT
            else:
                raise ResourceNotFound(path)
        return res

    def _make_fs_path(self, path):
        # type: (unicode) -> FSPath
        path = abspath(normpath(path))
        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            # synthetic root path
            return FSRootPath(self._wrap_fs, _ROOT, self._bundle)
        else:
            if not (parts[0] == self._bundle):
                # only files in the bundle exist in the filesystem
                raise ResourceNotFound(path)
            if l == 1:
                # /bundle
                return FSBundlePath(self._wrap_fs, path, self._bundle)
            elif l == 2:
                if _is_version_part(parts[1]):
                    # /bundle/version
                    return FSBundleVersionDirPath(self._wrap_fs, path)
                else:
                    # /bundle/collection
                    return FSCollectionPath(self._wrap_fs, path, self)
            elif l == 3:
                if _is_version_part(parts[1]):
                    # /bundle/version/file
                    return FSBundleVersionedFilePath(self._wrap_fs, path)
                elif _is_version_part(parts[2]):
                    # /bundle/collection/version
                    return FSCollectionVersionDirPath(self._wrap_fs, path)
                else:
                    # /bundle/collection/product
                    return FSProductPath(self._wrap_fs, path)
            elif l == 4:
                if _is_version_part(parts[2]):
                    # /bundle/collection/version/file
                    return FSCollectionVersionedFilePath(self._wrap_fs,
                                                         path, self)
                else:
                    # /bundle/collection/product/version
                    assert _is_version_part(parts[3]), path
                    return FSProductVersionDirPath(self._wrap_fs, path, self)
            elif l == 5:
                assert _is_version_part(parts[3])
                return FSProductVersionedFilePath(self._wrap_fs, path, self)
            else:
                assert False, path

    def _get_visit_info(self, collection_path):
        return self._wrap_fs.filterdir(
            collection_path,
            files=None, dirs=[_VISIT_DIR_PAT],
            exclude_dirs=None, exclude_files=[ALL_PAT])

    def _get_collections(self):
        return [info.name
                for info
                in self._wrap_fs.filterdir(_ROOT,
                                           files=None,
                                           dirs=['data_*'],
                                           exclude_dirs=None,
                                           exclude_files=[ALL_PAT])]

    def _get_collection_files(self, collection_path):
        # type: (unicode) -> List[Tuple[unicode, unicode]]
        return [(visit_info.name, file)
                for visit_info in self._get_visit_info(collection_path)
                for file in self._wrap_fs.listdir(join(collection_path,
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
        return [info.name for info in self._wrap_fs.filterdir(
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
                        raw_info = self._wrap_fs.getinfo(c).raw
            elif l == 3:
                _b, c, p = parts
                if _is_version_part(c):
                    # path = b/$n/file; orig path = /file
                    if p == _VERSION_DICT:
                        raw_info = _RAW_VERSION_DICT_INFO
                    else:
                        with unwrap_errors(path):
                            raw_info = self._wrap_fs.getinfo(p).raw
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
                            raw_info = self._wrap_fs.getinfo(join(c, f)).raw
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
                        raw_info = self._wrap_fs.getinfo(filepath).raw
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
            assert False, 'ArchiveToVersionedFS.openbin()'

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
                            for info in self._wrap_fs.scandir(_ROOT)
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
                    if self._wrap_fs.exists(p):
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
                    if self._wrap_fs.exists(join(c, f)):
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
    return ArchiveToVersionedFS(u'/Users/spaceman/Desktop/Archive/hst_14334')


if __name__ == '__main__':
    fs1 = ArchiveToVersionedFS(u'/Users/spaceman/Desktop/Archive/hst_14334')
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
