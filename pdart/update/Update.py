import abc

from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteFS
from pdart.fs.VersionView import VersionView

if TYPE_CHECKING:
    from typing import Callable
    from fs.base import FS
    from pdart.pds4.LIDVID import LIDVID


def update_bundle(versioned_fs, is_major, update):
    # type: (FS, bool, Callable[[CopyOnWriteFS], None]) -> None
    last_bundle_lidvid = versioned_fs.get_last_bundle_lidvid()
    version_view = VersionView(last_bundle_lidvid, versioned_fs)

    cow_fs = CopyOnWriteFS(version_view)
    update(cow_fs)
    cow_fs.normalize()

    delta = cow_fs.delta

    apply_delta(versioned_fs, is_major, delta)


def apply_delta(versioned_fs, is_major, delta):
    # type: (FS, bool, Delta) -> None
    assert False, 'apply_delta unimplemented'


class ISingleVersionBundleFS(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def bundle_lidvid(self):
        """Return the LIDVID for the bundle the filesystem holds."""
        # type: () -> LIDVID
        pass


class IMultiversionBundleFS(object):
    __metaclass__ = abc.ABCMeta

    # TODO This and things like the Version file need to go into a
    # single class down low.  With tests.  I can work on that.

    @abc.abstractmethod
    def lidvid_to_directory(self, lidvid):
        """For a given LIDVID, give the directory that contains its files."""
        # type: (LIDVID) -> unicode
        pass
