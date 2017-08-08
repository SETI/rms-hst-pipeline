from pdart.fs.DeletionPredicate import DeletionPredicate


class SetDeletionPredicate(DeletionPredicate):
    def __init__(self):
        self._deleted_paths = set()

    def is_deleted(self, path):
        return path in self._deleted_paths

    def delete(self, path):
        self._deleted_paths.add(path)
