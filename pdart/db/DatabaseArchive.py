import os.path
import sqlite3


class DatabaseArchive:
    """
    An object wrapper around a Sqlite connection to the archive's
    database.
    """
    def __init__(self, database_filepath):
        """
        Given the filepath to the database, open a connection and wrap
        it.  Note that the database will be created if no file exists
        at the filepath.
        """
        self.archive_dir = os.path.dirname(database_filepath)
        self.database_filepath = database_filepath
        self.connection = sqlite3.connect(database_filepath)

    def is_open(self):
        """Return True if the connection is open."""
        return self.connection is not None

    def close(self):
        """Close the connection.  If it's already closed, do nothing."""
        if self.is_open():
            self.connection.close()
            self.connection = None

    def execute_sql(self, sql, arg_tuple):
        """
        Given a SQL string possibly containing placeholders and a tuple of
        arguments, plug the arguments into the placeholder, execute the SQL
        and return a generator to the results.
        """
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(sql, arg_tuple)
            for res_tuple in cursor.fetchall():
                yield res_tuple

    def execute_sql_once(self, sql, arg_tuple):
        """
        Given a SQL string possibly containing placeholders and a
        tuple of arguments, plug the arguments into the placeholder,
        execute the SQL and return a single result.
        """
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(sql, arg_tuple)
            return cursor.fetchone()
