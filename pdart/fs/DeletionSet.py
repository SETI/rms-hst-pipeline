from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Set


class DeletionSet(object):
    def __init__(self):
        # type: () -> None
        self._deleted_paths = set()
        # type: Set[unicode]

    def is_deleted(self, path):
        # type: (unicode) -> bool
        return path in self._deleted_paths

    def delete(self, path):
        # type: (unicode) -> None
        self._deleted_paths.add(path)

    def undelete(self, path):
        # type: (unicode) -> None
        # raises KeyError in path not in set
        self._deleted_paths.remove(path)

    def as_set(self):
        # type: () -> Set[unicode]
        return self._deleted_paths
