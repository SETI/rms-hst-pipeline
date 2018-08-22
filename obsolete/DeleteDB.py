"""
***SCRIPT***  Delete the database of the archive.
"""
import os
import os.path

from pdart.pds4.Archives import *
from pdart.pds4labels.DBCalls import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.pds4.Archive import Archive


def _check_deletions(archive):
    # type: (Archive) -> None
    db_filepath = archive_database_filepath(archive)
    if os.path.isfile(db_filepath):
        assert not os.path.isfile(db_filepath)
    for bundle in archive.bundles():
        db_filepath = bundle_database_filepath(bundle)
        if os.path.isfile(db_filepath):
            assert not os.path.isfile(db_filepath)


if __name__ == '__main__':
    archive = get_any_archive()
    db_filepath = archive_database_filepath(archive)
    if os.path.isfile(db_filepath):
        os.remove(db_filepath)
    for bundle in archive.bundles():
        db_filepath = bundle_database_filepath(bundle)
        if os.path.isfile(db_filepath):
            os.remove(db_filepath)

    _check_deletions(archive)
