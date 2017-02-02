"""
**SCRIPT:** Run through the archive and extract information into a
SQLite database.
"""
import os.path
import sqlite3

from pdart.db.CreateDatabase import ArchiveDatabaseCreator
from pdart.pds4.Archives import *
from pdart.pds4labels.DBCalls import *


def run():
    # type: () -> None
    archive = get_any_archive()
    with closing(open_archive_database(archive)) as conn:
        ArchiveDatabaseCreator(conn, archive).create()

    db_filepath = archive_database_filepath(archive)
    assert os.path.isfile(db_filepath)  # a wimpy sanity check

    # dump it
    dump = False
    if dump:
        conn = sqlite3.connect(db_filepath)
        for line in conn.iterdump():
            print line

if __name__ == '__main__':
    run()
