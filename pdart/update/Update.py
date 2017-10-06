import abc

from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteFS, FSDelta
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView

if TYPE_CHECKING:
    from typing import Callable
    from fs.base import FS
    from pdart.pds4.LIDVID import LIDVID

    _UPDATE_FUNC = Callable[[CopyOnWriteFS], None]


def update_bundle(versioned_fs, is_major, update):
    # type: (MultiversionBundleFS, bool, _UPDATE_FUNC) -> None
    last_bundle_lidvid = versioned_fs.get_last_bundle_lidvid()
    version_view = VersionView(last_bundle_lidvid, versioned_fs)

    cow_fs = CopyOnWriteFS(version_view)
    update(cow_fs)
    cow_fs._remove_duplicates()

    delta = cow_fs.delta()

    apply_delta(versioned_fs, is_major, delta)


def apply_delta(versioned_fs, is_major, delta):
    # type: (FS, bool, FSDelta) -> None
    assert False, 'apply_delta unimplemented'


class ISingleVersionBundleFS(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def bundle_lidvid(self):
        """Return the LIDVID for the bundle the filesystem holds."""
        # type: () -> LIDVID
        pass
