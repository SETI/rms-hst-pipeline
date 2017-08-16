"""
Work on viewing an archive folder as a versioned filesystem.
"""
import io
import pickle
import sys
import traceback
from typing import TYPE_CHECKING

from fs.base import FS
from fs.errors import InvalidPath, ResourceNotFound
from fs.error_tools import unwrap_errors
from fs.info import Info
from fs.mode import check_writable
from fs.osfs import OSFS
from fs.path import abspath, basename, iteratepath, join, normpath

from pdart.fs.ReadOnlyView import ReadOnlyView

if TYPE_CHECKING:
    from typing import Tuple


_ROOT = u'/'


def _is_root(path):
    return abspath(normpath(path)) == _ROOT


def _is_version_part(part):
    return part.startswith(u'v$')

_VERSION_ONE = u'v$1'

_VISIT_DIR_PAT = u'visit_*'

_VERSION_DICT = u'version$dict'

ALL_PAT = u'*'

FILE_EXCLUSION_PATS = [u'.DS_Store', '*.db']

_RAW_VERSION_DICT_INFO = {u'basic': {u'name': _VERSION_DICT, u'is_dir': False}}


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

        def make_raw_dir_info(name):
            return {u'basic': {u'name': name, u'is_dir': True}}

        parts = iteratepath(path)
        l = len(parts)
        if l == 0:
            # path = /; purely synthetic
            raw_info = make_raw_dir_info(u'')
        else:
            if not (parts[0] == self._bundle):
                raise ResourceNotFound(path)
            if l == 1:
                # path = /b; orig path = /
                raw_info = make_raw_dir_info(self._bundle)
            elif l == 2:
                _b, c = parts
                if _is_version_part(c):
                    # path = b/$n
                    raw_info = make_raw_dir_info(c)
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
                    raw_info = make_raw_dir_info(p)
                else:
                    # path = b/c/p; synthetic
                    raw_info = make_raw_dir_info(p)
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
                    raw_info = make_raw_dir_info(f)
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
        self.check()
        if check_writable(mode):
            raise ResourceReadOnly(path)
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
    fs = ArchiveToVersionedFS(u'/Users/spaceman/Desktop/Archive/hst_14334')
    # import pudb; pu.db
    print "fs.listdir(u'/hst_14334') = %s" % list(fs.listdir(u'/hst_14334'))
    print "fs.scandir(u'/hst_14334') = %s" % list(fs.scandir(u'/hst_14334'))
    print "fs.filterdir(u'/hst_14334') = %s" % \
        list(fs.filterdir(u'/hst_14334'))
    fs.tree()
    # print fs._get_collection_fits_products(u'data_wfc3_raw')
    print fs._get_versions_dict(u'/hst_14334/v$1')
    print fs._get_versions_dict(u'/hst_14334/data_wfc3_raw/v$1')
    print fs
