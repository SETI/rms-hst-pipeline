"""
***SCRIPT***  Delete the database of the archive.
"""
import os
import os.path

from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archives import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.pds4.Archive import Archive


def _check_deletions(archive):
    # type: (Archive) -> None
    archive_dir = archive.root
    db_filepath = os.path.join(archive_dir, DATABASE_NAME)
    if os.path.isfile(db_filepath):
        assert not os.path.isfile(db_filepath)
    for bundle in archive.bundles():
        db_filepath = os.path.join(bundle.absolute_filepath(), DATABASE_NAME)
        if os.path.isfile(db_filepath):
            assert not os.path.isfile(db_filepath)


if __name__ == '__main__':
    archive = get_any_archive()
    archive_dir = archive.root
    db_filepath = os.path.join(archive_dir, DATABASE_NAME)
    if os.path.isfile(db_filepath):
        os.remove(db_filepath)
    for bundle in archive.bundles():
        db_filepath = os.path.join(bundle.absolute_filepath(), DATABASE_NAME)
        if os.path.isfile(db_filepath):
            os.remove(db_filepath)

    _check_deletions(archive)
