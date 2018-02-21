"""
An abstract base class marking a filesystem that contains a single
version of a single bundle.
"""
import abc

from fs.path import iteratepath
from typing import TYPE_CHECKING

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from pdart.pds4.VID import VID


class ISingleVersionBundleFS(object):
    """
    A filesystem that contains a single version of a single bundle.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def bundle_lidvid(self):
        """Return the LIDVID for the bundle the filesystem holds."""
        # type: () -> LIDVID
        pass

    @abc.abstractmethod
    def lid_to_vid(self, lid):
        # type: (LID) -> VID
        """
        Return the VID for the object corresponding to the given LID
        in the filesystem.
        """
        pass

    def lid_to_lidvid(self, lid):
        # type: (LID) -> LIDVID
        """
        Return the LIDVID for the object corresponding to the given
        LID in the filesystem.
        """
        return LIDVID.create_from_lid_and_vid(lid, self.lid_to_vid(lid))

    def directory_to_lidvid(self, dir):
        # type: (unicode) -> LIDVID
        """
        Return the LIDVID for the object that lives at the given
        directory path.
        """
        lid = self.directory_to_lid(dir)
        return LIDVID.create_from_lid_and_vid(lid, self.lid_to_vid(lid))

    @staticmethod
    def directory_to_lid(dir):
        # type: (unicode) -> LID
        """Return the LID corresponding to the given directory."""
        return LID.create_from_parts(iteratepath(dir))
