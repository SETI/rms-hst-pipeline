"""
**SCRIPT:** Experimentation using pyfilesystem.
"""

from fs.base import FS
import fs.compress
from fs.enums import ResourceType
from fs.error_tools import convert_os_errors
from fs.info import Info
from fs.mode import Mode, validate_open_mode
from fs.osfs import OSFS
from fs.permissions import Permissions
from fs.test import FSTestCases
from fs.wrapfs import WrapFS
import io
import nose
import os.path
import platform
import six
import stat


_WINDOWS_PLATFORM = platform.system() == 'Windows'


def _resource_is_hidden(rsrc):
    return rsrc[0] == '.' or rsrc.endswith('.db')


class Pds4FSR(FS):
    def __init__(self, root):
        FS.__init__(self)
        self._root = root

    def getsyspath(self, path):
        return os.path.join(self._root, path.lstrip('/'))

    @classmethod
    def _make_access_from_stat(cls, stat_result):
        access = {}
        access['permissions'] = Permissions(
            mode=stat_result.st_mode
        ).dump()
        access['gid'] = stat_result.st_gid
        access['uid'] = stat_result.st_uid
        if not _WINDOWS_PLATFORM:
            import grp
            import pwd
            try:
                access['group'] = grp.getgrgid(access['gid']).gr_name
            except KeyError:  # pragma: nocover
                pass

            try:
                access['user'] = pwd.getpwuid(access['uid']).pw_name
            except KeyError:  # pragma: nocover
                pass
        return access

    STAT_TO_RESOURCE_TYPE = {
        stat.S_IFDIR: ResourceType.directory,
        stat.S_IFCHR: ResourceType.character,
        stat.S_IFBLK: ResourceType.block_special_file,
        stat.S_IFREG: ResourceType.file,
        stat.S_IFIFO: ResourceType.fifo,
        stat.S_IFLNK: ResourceType.symlink,
        stat.S_IFSOCK: ResourceType.socket
    }

    @classmethod
    def _get_type_from_stat(cls, _stat):
        """Get the resource type from a stat_result object."""
        st_mode = _stat.st_mode
        st_type = stat.S_IFMT(st_mode)
        return cls.STAT_TO_RESOURCE_TYPE.get(st_type, ResourceType.unknown)

    @classmethod
    def _make_details_from_stat(cls, stat_result):
        """Make an info dict from a stat_result object."""
        details = {
            '_write': ['accessed', 'modified'],
            'accessed': stat_result.st_atime,
            'modified': stat_result.st_mtime,
            'size': stat_result.st_size,
            'type': int(cls._get_type_from_stat(stat_result))
        }
        # On other Unix systems (such as FreeBSD), the following
        # attributes may be available (but may be only filled out if
        # root tries to use them):
        details['created'] = getattr(stat_result, 'st_birthtime', None)
        ctime_key = (
            'created'
            if _WINDOWS_PLATFORM
            else 'metadata_changed'
        )
        details[ctime_key] = stat_result.st_ctime
        return details

    # reading
    def getinfo(self, path, namespaces=None):
        self.check()
        namespaces = namespaces or ()
        sys_path = self.getsyspath(path)

        with convert_os_errors('getinfo', path):
            _stat = os.stat(sys_path)

        info = {
            'basic': {
                'name': os.path.basename(path),
                'is_dir': stat.S_ISDIR(_stat.st_mode)
            }
        }
        if 'details' in namespaces:
            info['details'] = self._make_details_from_stat(_stat)
        if 'stat' in namespaces:
            info['stat'] = {
                k: getattr(_stat, k)
                for k in dir(_stat) if k.startswith('st_')
            }
        if 'access' in namespaces:
            info['access'] = self._make_access_from_stat(_stat)

        return Info(info)

    def listdir(self, path):
        self.check()

        sys_path = self.getsyspath(path)
        # print 'sys_path =', sys_path
        with convert_os_errors('listdir', path, directory=True):
            names = os.listdir(sys_path)
            # print 'names =', names
        return [rsrc for rsrc in names if not _resource_is_hidden(rsrc)]

    def openbin(self, path, mode='r', buffering=-1, **options):
        _mode = Mode(mode)
        _mode.validate_bin()
        self.check()
        sys_path = self.getsyspath(path)
        with convert_os_errors('openbin', path):
            if six.PY2 and _mode.exclusive and self.exists(path):
                raise errors.FileExists(path)
            binary_file = io.open(
                sys_path,
                mode=_mode.to_platform_bin(),
                buffering=buffering,
                **options
            )
        return binary_file

    # writing
    def makedir(self, path, permissions=None, recreate=False):
        assert False, 'makedir unimplemented'

    def remove(self, path):
        assert False, 'remove unimplemented'

    def removedir(self, path):
        assert False, 'removedir unimplemented'

    def setinfo(self, path, info):
        assert False, 'setinfo unimplemented'


class Pds4FSRTestCases(FSTestCases):
    def make_fs(self):
        return Pds4FSR(ROOT)


ROOT = u'/Users/spaceman/Desktop/Archive/hst_05167'


def run():
    # pfs = Pds4FSW(ROOT)
    pfs = Pds4FSR(ROOT)
    print 'Hi, from Filesystem!'
    pfs.tree()
    fs.compress.write_tar(pfs, 'foo.tar')


if __name__ == '__main__':
    run()
    # nose.run()
