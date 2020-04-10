"""
This module provides an object-oriented wrapper around a SQLite
connection.

*This module is currently unused (as of 2016-08-24).*
"""
from contextlib import closing
import sqlite3

from fs.path import dirname

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Iterable, Iterator, Tuple


class DatabaseArchive:
    """
    An object wrapper around a SQLite connection to the archive's
    database.
    """
    def __init__(self, database_filepath):
        # type: (unicode) -> None
        """
        Given the filepath to the database, open a connection and wrap
        it.  Note that the database will be created if no file exists
        at the filepath.
        """
        self.archive_dir = dirname(database_filepath)
        self.database_filepath = database_filepath
        self.connection = sqlite3.connect(database_filepath)

    def is_open(self):
        # type: () -> bool
        """Return True if the connection is open."""
        return self.connection is not None

    def close(self):
        # type: () -> None
        """Close the connection.  If it's already closed, do nothing."""
        if self.is_open():
            self.connection.close()
            self.connection = None

    def execute_sql(self, sql, arg_tuple):
        # type: (str, Iterable[Any]) -> Iterator[Tuple[Any, ...]]
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
        # type: (str, Iterable[Any]) -> Tuple[Any, ...]
        """
        Given a SQL string possibly containing placeholders and a
        tuple of arguments, plug the arguments into the placeholder,
        execute the SQL and return a single result.
        """
        with closing(self.connection.cursor()) as cursor:
            cursor.execute(sql, arg_tuple)
            return cursor.fetchone()
