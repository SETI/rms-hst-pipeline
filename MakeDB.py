"""
**SCRIPT:** Run through the archive and extract information into a
SQLite database.
"""
import os.path
import sqlite3

from pdart.db.CreateDatabase import create_database
from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archives import *

if __name__ == '__main__':
    archive = get_any_archive()
    archive_dir = get_any_archive_dir()
    db_filepath = os.path.join(archive_dir, DATABASE_NAME)
    conn = sqlite3.connect(db_filepath)
    try:
        create_database(conn, archive)
    finally:
        conn.close()

    assert os.path.isfile(db_filepath)  # a wimpy sanity check

    # dump it
    dump = False
    if dump:
        conn = sqlite3.connect(db_filepath)
        for line in conn.iterdump():
            print line
        conn.close()
