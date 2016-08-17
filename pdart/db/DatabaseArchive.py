import sqlite3


class DatabaseArchive:
    def __init__(self, database_filepath):
        self.database_filepath = database_filepath
        self.connection = sqlite3.connect(database_filepath)

    def is_open(self):
        return self.connection is not None

    def close(self):
        if self.is_open():
            self.connection.close()
            self.connection = None
