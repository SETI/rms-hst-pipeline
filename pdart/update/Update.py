import abc

from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteFS, FSDelta
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView

if TYPE_CHECKING:
    from typing import Callable
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
    # type: (MultiversionBundleFS, bool, FSDelta) -> None
    dirs = delta.directories()
    if not dirs:
        return

    # Note that these directories are from an unversioned view.
    create_new_version_directories(versioned_fs, is_major, dirs)
    populate_new_version_directories(versioned_fs, is_major, dirs, delta)
    assert False, 'apply_delta() unimplemented'


def create_new_version_directories(versioned_fs, is_major, dirs):
    # type: (MultiversionBundleFS, bool, List[unicode]) -> None

    # Note that these directories are from an unversioned view.
    for dir in dirs:
        lid = dir_to_lid(dir)

    assert False, 'create_new_version_directories() unimplemented'


def populate_new_version_directories(versioned_fs, is_major, dirs, delta):
    # type: (MultiversionBundleFS, bool, List[unicode], FSDelta) -> None

    # Note that these directories are from an unversioned view.
    assert False, 'populate_new_version_directories() unimplemented'


def dir_to_lid(lid):
    # type: (LID)-> unicode
    assert False, 'dir_to_lid() unimplemented'


class ISingleVersionBundleFS(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def bundle_lidvid(self):
        """Return the LIDVID for the bundle the filesystem holds."""
        # type: () -> LIDVID
        pass
