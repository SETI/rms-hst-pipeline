from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteFS
from pdart.fs.VersionView import VersionView

if TYPE_CHECKING:
    from typing import Callable
    from fs.base import FS


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
