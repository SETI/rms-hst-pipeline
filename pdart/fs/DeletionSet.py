"""
A utility class to track changes in a copy-on-write filesystem.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Set


class DeletionSet(object):
    """
    Represents a set of paths to files and directories that have been
    deleted from a copy-on-write filesystem.
    """
    def __init__(self):
        # type: () -> None
        self._deleted_paths = set()
        # type: Set[unicode]

    def __str__(self):
        return 'DeletionSet(%s)' % self._deleted_paths

    def is_deleted(self, path):
        # type: (unicode) -> bool
        """
        Return True if the file or directory at the given path was
        deleted.
        """
        return path in self._deleted_paths

    def delete(self, path):
        # type: (unicode) -> None
        """
        Mark the file or directory at the given path as deleted.
        """
        self._deleted_paths.add(path)

    def undelete(self, path):
        # type: (unicode) -> None
        # raises KeyError if path not in set
        """
        Unmark the file or directory at the given path as deleted.
        """
        self._deleted_paths.remove(path)

    def as_set(self):
        # type: () -> Set[unicode]
        """
        Return the set of deleted paths as a Python set.
        """
        return self._deleted_paths
