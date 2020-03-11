"""
The functionality to update a bundle.
"""
from fs.copy import copy_file
from fs.path import join, split, splitext
from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteVersionView
from pdart.fs.DirUtils import lid_to_dir, lidvid_to_dir
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.OldVersionView import OldVersionView
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Callable

    # some type aliases
    _COWVV = CopyOnWriteVersionView
    FILENAME_FILTER = Callable[[unicode], bool]
    _MBFS = MultiversionBundleFS
    _UPDATE_FUNC = Callable[[CopyOnWriteVersionView], None]
    _LIDVID_INCR = Callable[[LIDVID], LIDVID]


def lidvid_incrementor(is_major):
    # type: (bool) ->  _LIDVID_INCR
    """
    Return a function that increments a LIDVID.  If is_major is true,
    the major version will be incremented, else the minor version.
    """
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
    """Return true if the filepath is to a FITS file."""
    (_, ext) = splitext(path)
    return ext == '.fits'


def update_bundle(multiversioned_fs, last_bundle_lidvid, is_major, update):
    # type: (MultiversionBundleFS, LIDVID, bool, _UPDATE_FUNC) -> _COWVV
    """
    Use the update function (or callable) to update the bundle with
    the given LIDVID in the filesystem.
    """
    last_version_view = OldVersionView(last_bundle_lidvid, multiversioned_fs)

    cow_fs = CopyOnWriteVersionView(last_version_view)
    update(cow_fs)
    cow_fs.normalize()

    # TODO I also need to remove derived files (like labels) and
    # regenerate them.

    delta = cow_fs.delta()

    # Note that these directories are from a single-version view.
    dirs = delta.directories()
    if not dirs:
        return cow_fs

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

        # TODO Rethink and clarify the semantics of dirs.  They play
        # multiple roles.
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
            path_dir, base = split(path)
            old_lidvid = cow_fs.directory_to_lidvid(path_dir)
            new_lidvid = incr_lidvid(old_lidvid)
            multiversioned_fs_path = join(lidvid_to_dir(new_lidvid), base)
            copy_file(cow_fs, path, multiversioned_fs, multiversioned_fs_path)

    bundle_dir = lid_to_dir(last_bundle_lidvid.lid())
    copy_directory_at(bundle_dir)
    return cow_fs
