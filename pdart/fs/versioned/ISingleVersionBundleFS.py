"""
An abstract base class marking a filesystem that contains a single
version of a single bundle.
"""
import abc

from fs.path import iteratepath

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID


class ISingleVersionBundleFS(object, metaclass=abc.ABCMeta):
    """
    A filesystem that contains a single version of a single bundle.
    """

    def is_single_versioned_fs(self) -> bool:
        return True

    @abc.abstractmethod
    def bundle_lidvid(self) -> LIDVID:
        """Return the LIDVID for the bundle the filesystem holds."""
        pass

    @abc.abstractmethod
    def lid_to_vid(self, lid: LID) -> VID:
        """
        Return the VID for the object corresponding to the given LID
        in the filesystem.
        """
        pass

    def lid_to_lidvid(self, lid: LID) -> LIDVID:
        """
        Return the LIDVID for the object corresponding to the given
        LID in the filesystem.
        """
        return LIDVID.create_from_lid_and_vid(lid, self.lid_to_vid(lid))

    def directory_to_lidvid(self, dir: str) -> LIDVID:
        """
        Return the LIDVID for the object that lives at the given
        directory path.
        """
        lid = self.directory_to_lid(dir)
        return LIDVID.create_from_lid_and_vid(lid, self.lid_to_vid(lid))

    @staticmethod
    def directory_to_lid(dir: str) -> LID:
        """Return the LID corresponding to the given directory."""
        return LID.create_from_parts([str(part) for part in iteratepath(dir)])
