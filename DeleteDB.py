"""
***SCRIPT***  Delete the database of the archive.
"""
import os
import os.path

from pdart.pds4.Archives import *

if __name__ == '__main__':
    archive_dir = get_any_archive_dir()
    db_filepath = os.path.join(archive_dir, 'archive.spike.db')
    if os.path.isfile(db_filepath):
        os.remove(db_filepath)
