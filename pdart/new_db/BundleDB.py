import sqlite3

_BUNDLE_DB_NAME = 'bundle$database.db'


# type: unicode

class BundleDB(object):
    def __init__(self, filepath):
        # type: (unicode) -> None
        self.filepath = filepath
        self.connection = sqlite3.connect(str(filepath))

    def close(self):
        # type: () -> None
        self.connection.close()
        self.connection = None

    def is_open(self):
        # type: () -> Bool
        return self.connection is not None
