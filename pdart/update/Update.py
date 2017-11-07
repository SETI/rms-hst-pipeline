from fs.copy import copy_file
from fs.path import join, split, splitext
from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteVersionView
from pdart.fs.DirUtils import lid_to_dir, lidvid_to_dir
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView

if TYPE_CHECKING:
    from typing import Callable
    from pdart.pds4.LIDVID import LIDVID

    # from pdart.pds4.VID import VID

    # some type aliases
    _COWVV = CopyOnWriteVersionView
    _FILENAME_FILTER = Callable[[unicode], bool]
    _MBFS = MultiversionBundleFS
    _UPDATE_FUNC = Callable[[CopyOnWriteVersionView], None]
    _LIDVID_INCR = Callable[[LIDVID], LIDVID]


def lidvid_incrementor(is_major):
    # type: (bool) ->  _LIDVID_INCR
    def major_increment(lidvid):
        # type: (LIDVID) -> LIDVID
        return lidvid.next_major_lidvid()

    def minor_increment(lidvid):
        # type: (LIDVID) -> LIDVID
        return lidvid.next_minor_lidvid()

    if is_major:
        return major_increment
    else:
        return minor_increment


def is_fits_file(path):
    # type: (unicode) -> bool
    (_, ext) = splitext(path)
    return ext == '.fits'


def update_bundle(multiversioned_fs, last_bundle_lidvid, is_major, update):
    # type: (MultiversionBundleFS, LIDVID, bool, _UPDATE_FUNC) -> _COWVV
    last_version_view = VersionView(last_bundle_lidvid, multiversioned_fs)

    cow_fs = CopyOnWriteVersionView(last_version_view)
    update(cow_fs)
    cow_fs._remove_duplicates()

    delta = cow_fs.delta()

    # Note that these directories are from a single-version view.
    dirs = delta.directories()
    if not dirs:
        return

    incr_lidvid = lidvid_incrementor(is_major)

    # copy files here
    def copy_entry_at(path):
        # type: (unicode) -> None
        if cow_fs.isdir(path):
            copy_directory_at(path)
        elif cow_fs.isfile(path):
            copy_file_at(path)
        else:
            assert False, 'impossible case at %s' % path

    def new_parent_lidvid(lidvid):
        # type: (LIDVID) -> LIDVID
        parent_lid = lidvid.lid().parent_lid()
        old_parent_lidvid = cow_fs.lid_to_lidvid(parent_lid)
        return incr_lidvid(old_parent_lidvid)

    def copy_directory_at(path):
        # type: (unicode) -> None
        old_lidvid = cow_fs.directory_to_lidvid(path)
        new_lidvid = incr_lidvid(old_lidvid)
        if path not in dirs:
            # it was not changed, but we need to include it as a child
            multiversioned_fs.add_subcomponent(new_parent_lidvid(new_lidvid),
                                               new_lidvid)
        # handle changed directory
        elif new_lidvid.is_bundle_lidvid():
            multiversioned_fs.make_lidvid_directories(new_lidvid)
        else:
            multiversioned_fs.add_subcomponent(new_parent_lidvid(new_lidvid),
                                               new_lidvid)

        for entry in cow_fs.listdir(path):
            child_path = join(path, entry)
            copy_entry_at(child_path)

    def copy_file_at(path):
        # type: (unicode) -> None
        if is_fits_file(path):
            # this is the version-less path
            cow_fs_path = path
            dir, base = split(cow_fs_path)
            old_lidvid = cow_fs.directory_to_lidvid(dir)
            new_lidvid = incr_lidvid(old_lidvid)
            multiversioned_fs_path = join(lidvid_to_dir(new_lidvid), base)
            copy_file(cow_fs, path, multiversioned_fs, multiversioned_fs_path)

    bundle_dir = lid_to_dir(last_bundle_lidvid.lid())
    copy_directory_at(bundle_dir)
    return cow_fs
