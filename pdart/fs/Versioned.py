from typing import TYPE_CHECKING
import abc
from fs.osfs import OSFS
from cowfs.COWFS import COWFS

if TYPE_CHECKING:
    from typing import Optional
    from fs.base import FS

class Versioned(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_single_versioned_fs(self):
        # type: () -> bool
        pass

    def is_multiversioned_fs(self):
        # type: () -> bool
        return not self.is_single_versioned_fs()

############################################################

class SingleVersionedOSFS(OSFS, Versioned):
    def __init__(self, root_path, create=False,
                 create_mode=511, expand_vars=True):
        # type: (unicode, bool, int, bool) -> None
        OSFS.__init__(self, root_path, create, create_mode, expand_vars)

    def is_single_versioned_fs(self):
        # type: () -> bool
        return True

    @staticmethod
    def create_suffixed(partial_root_path, create=False,
                        create_mode=511, expand_vars=True):
        # type: (unicode, bool, int, bool) -> SingleVersionedOSFS
        return SingleVersionedOSFS(partial_root_path + '-sv',
                                   create,
                                   create_mode,
                                   expand_vars)


class MultiversionedOSFS(OSFS, Versioned):
    def __init__(self, root_path, create=False,
                 create_mode=511, expand_vars=True):
        # type: (unicode, bool, int, bool) -> None
        OSFS.__init__(self, root_path, create, create_mode, expand_vars)

    def is_single_versioned_fs(self):
        # type: () -> bool
        return False

    @staticmethod
    def create_suffixed(partial_root_path, create=False,
                        create_mode=511, expand_vars=True):
        # type: (unicode, bool, int, bool) -> MultiversionedOSFS
        return MultiversionedOSFS(partial_root_path + '-mv',
                                  create,
                                  create_mode,
                                  expand_vars)

############################################################

class SingleVersionedCOWFS(COWFS, Versioned):
    def __init__(self, base_fs, additions_fs=None, deletions_fs=None):
        # type: (FS, Optional[FS], Optional[FS]) -> None
        assert isinstance(base_fs, Versioned)
        assert base_fs.is_single_versioned_fs()
        COWFS.__init__(self, base_fs, additions_fs, deletions_fs)

    def is_single_versioned_fs(self):
        # type: () -> bool
        return True

    @staticmethod
    def create_cowfs_suffixed(base_fs,
                              deltas_layer_partial_path,
                              recreate=False):
        # type: (FS, unicode, bool) -> COWFS
        deltas_layer_path = \
            deltas_layer_partial_path + u'-deltas-layer-sv'
        rwfs = SingleVersionedOSFS(deltas_layer_path, create=recreate)
        additions_fs = rwfs.makedir(u'/additions', recreate=recreate)
        deletions_fs = rwfs.makedir(u'/deletions', recreate=recreate)
        return SingleVersionedCOWFS(rwfs,
                                    additions_fs,
                                    deletions_fs)

class MultiversionedCOWFS(COWFS, Versioned):
    def __init__(self, base_fs, additions_fs=None, deletions_fs=None):
        # type: (FS, Optional[FS], Optional[FS]) -> None
        assert isinstance(base_fs, Versioned)
        assert base_fs.is_multiversioned_fs()
        COWFS.__init__(self, base_fs, additions_fs, deletions_fs)

    def is_single_versioned_fs(self):
        # type: () -> bool
        return False

    @staticmethod
    def create_cowfs_suffixed(base_fs,
                              deltas_layer_partial_path,
                              recreate=False):
        # type: (FS, unicode, bool) -> COWFS
        deltas_layer_path = \
            deltas_layer_partial_path + '-deltas-layer-mv'
        rwfs = MultiversionedOSFS(deltas_layer_path, create=recreate)
        additions_fs = rwfs.makedir(u'/additions', recreate=recreate)
        deletions_fs = rwfs.makedir(u'/deletions', recreate=recreate)
        return MultiversionedCOWFS(rwfs,
                                   additions_fs,
                                   deletions_fs)

