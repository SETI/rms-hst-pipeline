from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteVersionView
from pdart.fs.DirUtils import lid_to_dir
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView

if TYPE_CHECKING:
    from typing import Callable
    from pdart.pds4.LIDVID import LIDVID

    _UPDATE_FUNC = Callable[[CopyOnWriteVersionView], None]


def update_bundle(multiversioned_fs, last_bundle_lidvid, is_major, update):
    # type: (MultiversionBundleFS, LIDVID, bool, _UPDATE_FUNC) -> None
    last_version_view = VersionView(last_bundle_lidvid, multiversioned_fs)

    cow_fs = CopyOnWriteVersionView(last_version_view)
    update(cow_fs)
    cow_fs._remove_duplicates()

    delta = cow_fs.delta()

    dirs = delta.directories()
    if not dirs:
        return

    # Note that these directories are from an unversioned view.
    lidvids = map(lambda (d): multiversioned_fs.directory_to_lidvids(d), dirs)
    if is_major:
        new_lidvids = map(lambda (lv): lv.next_major_lidvid(), lidvids)
    else:
        new_lidvids = map(lambda (lv): lv.next_minor_lidvid(), lidvids)
    for new_lidvid in new_lidvids:
        multiversioned_fs.make_lidvid_directories(new_lidvid)

    old_and_new_lidvids = zip(lidvids, new_lidvids)
    old_to_new_lidvids = dict(old_and_new_lidvids)

    new_bundle_lidvid = old_to_new_lidvids[last_bundle_lidvid]
    new_version_view = VersionView(new_bundle_lidvid, multiversioned_fs)

    for (old_lidvid, new_lidvid) in old_and_new_lidvids:
        old_dir = lid_to_dir(old_lidvid.LID)
        new_dir = lid_to_dir(new_lidvid.LID)
        copy_selected_files(last_version_view, old_dir,
                            new_version_view, new_dir)

    # add labels and other metadata
    assert False, 'apply_delta() unimplemented'


def copy_selected_files(old_dir, new_dir):
    # type: (VersionView, unicode, VersionView, unicode) -> None
    assert False, "unimplemented"
