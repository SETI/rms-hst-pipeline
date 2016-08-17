import os.path
import sqlite3


class DatabaseArchive:
    def __init__(self, database_filepath):
        self.archive_dir = os.path.dirname(database_filepath)
        self.database_filepath = database_filepath
        self.connection = sqlite3.connect(database_filepath)

    def is_open(self):
        return self.connection is not None

    def close(self):
        if self.is_open():
            self.connection.close()
            self.connection = None

    def execute_sql(self, sql, arg_tuple):
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(sql, arg_tuple)
            for res_tuple in cursor.fetchall():
                yield res_tuple

    def execute_sql_once(self, sql, arg_tuple):
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(sql, arg_tuple)
            return cursor.fetchone()
