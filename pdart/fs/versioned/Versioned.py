import abc
import os
import os.path
from typing import Optional

from fs.base import FS
from fs.osfs import OSFS
from fs.tempfs import TempFS

from pdart.fs.cowfs.COWFS import COWFS


class Versioned(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_single_versioned_fs(self) -> bool:
        pass

    def is_multiversioned_fs(self) -> bool:
        return not self.is_single_versioned_fs()


############################################################


class SingleVersionedOSFS(OSFS, Versioned):
    def __init__(
        self,
        root_path: str,
        create: bool = False,
        create_mode: int = 511,
        expand_vars: bool = True,
    ) -> None:
        OSFS.__init__(self, root_path, create, create_mode, expand_vars)

    def is_single_versioned_fs(self) -> bool:
        return True

    @staticmethod
    def create_suffixed(
        partial_root_path: str,
        create: bool = False,
        create_mode: int = 511,
        expand_vars: bool = True,
    ) -> "SingleVersionedOSFS":
        root_path = partial_root_path + "-sv"
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
        return SingleVersionedOSFS(root_path, create, create_mode, expand_vars)


class MultiversionedOSFS(OSFS, Versioned):
    def __init__(
        self,
        root_path: str,
        create: bool = False,
        create_mode: int = 511,
        expand_vars: bool = True,
    ) -> None:
        OSFS.__init__(self, root_path, create, create_mode, expand_vars)

    def is_single_versioned_fs(self) -> bool:
        return False

    @staticmethod
    def create_suffixed(
        partial_root_path: str,
        create: bool = False,
        create_mode: int = 511,
        expand_vars: bool = True,
    ) -> "MultiversionedOSFS":
        root_path = partial_root_path + "-mv"
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
        return MultiversionedOSFS(root_path, create, create_mode, expand_vars)


############################################################


class SingleVersionedCOWFS(COWFS, Versioned):
    def __init__(
        self,
        base_fs: FS,
        additions_fs: Optional[FS] = None,
        deletions_fs: Optional[FS] = None,
    ) -> None:
        # TODO Lots of temporary hacks (1) until I bring the external
        # filesystems into the pdart tree, and (2) until I define a
        # SingleVersionedTempFS.  Get rid of them.  Ugly, ugly, ugly.
        from pdart.fs.versioned.ISingleVersionBundleFS import ISingleVersionBundleFS

        assert (
            isinstance(base_fs, Versioned)
            or isinstance(base_fs, ISingleVersionBundleFS)
            or isinstance(base_fs, TempFS)
        ), type(base_fs)
        assert isinstance(base_fs, TempFS) or base_fs.is_single_versioned_fs()
        COWFS.__init__(self, base_fs, additions_fs, deletions_fs)

    def is_single_versioned_fs(self) -> bool:
        return True

    @staticmethod
    def create_cowfs_suffixed(
        base_fs: FS, deltas_layer_partial_path: str, recreate: bool = False
    ) -> "SingleVersionedCOWFS":
        deltas_layer_path = deltas_layer_partial_path + "-deltas-sv"
        if not os.path.isdir(deltas_layer_path):
            os.makedirs(deltas_layer_path)
        rwfs = SingleVersionedOSFS(deltas_layer_path, create=recreate)
        additions_fs = rwfs.makedir("/additions", recreate=recreate)
        deletions_fs = rwfs.makedir("/deletions", recreate=recreate)
        return SingleVersionedCOWFS(base_fs, additions_fs, deletions_fs)


class MultiversionedCOWFS(COWFS, Versioned):
    def __init__(
        self,
        base_fs: FS,
        additions_fs: Optional[FS] = None,
        deletions_fs: Optional[FS] = None,
    ) -> None:
        assert isinstance(base_fs, Versioned)
        assert base_fs.is_multiversioned_fs()
        COWFS.__init__(self, base_fs, additions_fs, deletions_fs)

    def is_single_versioned_fs(self) -> bool:
        return False

    @staticmethod
    def create_cowfs_suffixed(
        base_fs: FS, deltas_layer_partial_path: str, recreate: bool = False
    ) -> "MultiversionedCOWFS":
        deltas_layer_path = deltas_layer_partial_path + "-deltas-mv"
        if not os.path.isdir(deltas_layer_path):
            os.makedirs(deltas_layer_path)
        rwfs = MultiversionedOSFS(deltas_layer_path, create=recreate)
        additions_fs = rwfs.makedir("/additions", recreate=recreate)
        deletions_fs = rwfs.makedir("/deletions", recreate=recreate)
        return MultiversionedCOWFS(base_fs, additions_fs, deletions_fs)
