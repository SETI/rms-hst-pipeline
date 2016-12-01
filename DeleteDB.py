"""
***SCRIPT***  Delete the database of the archive.
"""
import os
import os.path

from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archives import *


def _check_deletion(archive_dir):
    assert not os.path.isfile(os.path.join(archive_dir, DATABASE_NAME))


if __name__ == '__main__':
    archive_dir = get_any_archive_dir()
    db_filepath = os.path.join(archive_dir, DATABASE_NAME)
    if os.path.isfile(db_filepath):
        os.remove(db_filepath)
    _check_deletion(archive_dir)
