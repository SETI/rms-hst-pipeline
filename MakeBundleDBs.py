"""
**SCRIPT:** Run through the archive and extract information into a
SQLite database.
"""
import os.path

from pdart.db.CreateBundleDatabase import BundleDatabaseCreator
from pdart.pds4.Archives import *
from pdart.pds4labels.DBCalls import *


def run():
    # type: () -> None
    archive = get_any_archive()
    BundleDatabaseCreator(archive).create()

    for bundle in archive.bundles():
        db_filepath = bundle_database_filepath(bundle)
        assert os.path.isfile(db_filepath)  # a wimpy sanity check


if __name__ == '__main__':
    run()
