import abc


class DeletionPredicate(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_deleted(self, path):
        # type: (unicode) -> bool
        pass
