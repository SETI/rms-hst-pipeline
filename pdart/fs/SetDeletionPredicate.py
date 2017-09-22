from typing import TYPE_CHECKING

from pdart.fs.DeletionPredicate import DeletionPredicate

if TYPE_CHECKING:
    from typing import Set


class SetDeletionPredicate(DeletionPredicate):
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
