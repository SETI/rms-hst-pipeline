"""
SCRIPT: Run through the archive and extract information into a SQLite
database.
"""
from pdart.db.CreateDatabase import createDatabase

if __name__ == '__main__':
    archive = get_any_archive()
    archive_dir = archive.root
    db_filepath = os.path.join(archive_dir, 'archive.spike.db')
    conn = sqlite3.connect(db_filepath)
    try:
        createDatabase(conn, archive)
    finally:
        conn.close()

    # dump it
    dump = False
    if dump:
        conn = sqlite3.connect(db_filepath)
        for line in conn.iterdump():
            print line
        conn.close()
