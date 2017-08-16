"""
Work on viewing an archive folder as a versioned filesystem.
"""
import sys
import traceback
from typing import TYPE_CHECKING

from fs.base import FS
from fs.errors import InvalidPath, ResourceNotFound
from fs.error_tools import unwrap_errors
from fs.info import Info
from fs.osfs import OSFS
from fs.path import basename, iteratepath, join

from pdart.fs.ReadOnlyView import ReadOnlyView

if TYPE_CHECKING:
    from typing import Tuple


def _is_root(path):
    return abspath(normpath(path)) == u'/'


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
                    res = self._wrap_fs, u'/'
            else:
                raise ResourceNotFound(path)
        return res

    def getinfo(self, path, namespaces=None):
        self.check()
        _fs, _path = self._delegate_file_path(path)
        if _fs:
            with unwrap_errors(path):
                raw_info = _fs.getinfo(_path, namespaces=namespaces).raw
                if _is_root(_path):
                    raw_info[u'basic'][u'name'] = self._bundle
        else:
            raw_info = {u'basic': {u'name': u'', u'is_dir': True}}
        return Info(raw_info)

    def openbin(self, path, mode="r", buffering=-1, **options):
        self.check()
        # TODO need to raise on writing since we're R/O
        _fs, _path = self._delegate_file_path(path)
        if _fs:
            with unwrap_errors(path):
                bin_file = _fs.openbin(_path, mode=mode,
                                       buffering=buffering, **options)
        else:
            assert False, 'ArchiveToVersionedFS.openbin()'

    def listdir(self, path):
        self.check()
        _fs, _path = self._delegate_file_path(path)
        with unwrap_errors(path):
            if _fs:
                dir_list = _fs.listdir(_path)
            else:
                # handle root
                dir_list = [self._bundle]
        return dir_list


def test_fs():
    return ArchiveToVersionedFS(u'/Users/spaceman/Desktop/Archive/hst_09678')


if __name__ == '__main__':
    fs = ArchiveToVersionedFS(u'/Users/spaceman/Desktop/Archive/hst_09678')
    # import pudb; pu.db
    print "fs.listdir(u'/') = %s" % list(fs.listdir(u'/'))
    print "fs.scandir(u'/') = %s" % list(fs.scandir(u'/'))
    print "fs.filterdir(u'/') = %s" % list(fs.filterdir(u'/'))
    fs.tree()
    print fs
