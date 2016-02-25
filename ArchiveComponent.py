import abc
import os
import os.path

import ArchiveFile


class ArchiveComponent(object):
    """
    A bundle, component, or product within an archive.  This is an
    abstract class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, arch, lid):
        """
        Create an ArchiveComponent given the archive it lives in and
        its LID.
        """
        assert arch
        self.archive = arch
        assert lid
        self.lid = lid

    def __eq__(self, other):
        return self.archive == other.archive and self.lid == other.lid

    def __str__(self):
        return str(self.lid)

    @abc.abstractmethod
    def directory_filepath(self):
        """Return the absolute filepath to the component's directory."""
        pass

    def files(self):
        """
        Generate all the files belonging to this component as ArchiveFile
        objects.
        """
        dir = self.directory_filepath()
        for basename in os.listdir(dir):
            if basename[0] != '.':
                file = os.path.join(dir, basename)
                if (os.path.isfile(file)):
                    yield ArchiveFile.ArchiveFile(self, basename)
