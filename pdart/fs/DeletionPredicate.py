from abc import ABCMeta, abstractmethod


class DeletionPredicate(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def is_deleted(self, path):
        # type: (unicode) -> bool
        pass
