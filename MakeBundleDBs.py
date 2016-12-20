"""
**SCRIPT:** Run through the archive and extract information into a
SQLite database.
"""
import os.path
import sqlite3

from pdart.db.CreateBundleDatabase import BundleDatabaseCreator
from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archives import *


def run():
    # type: () -> None
    archive = get_any_archive()
    BundleDatabaseCreator(archive).create()

    for bundle in archive.bundles():
        bundle_path = bundle.absolute_filepath()
        db_filepath = os.path.join(bundle_path, DATABASE_NAME)
        assert os.path.isfile(db_filepath)  # a wimpy sanity check

if __name__ == '__main__':
    run()
