"""
Representation of a PDS4 bundle, collection, or product.
"""
import abc

from fs.path import join, splitext
from typing import TYPE_CHECKING

from pdart.pds4.LID import LID

if TYPE_CHECKING:
    from typing import Iterator
    import pdart.pds4.Archive
    import pdart.pds4.File


class Component(object):
    """
    A :class:`~pdart.pds4.Bundle`, :class:`~pdart.pds4.Collection`, or
    :class:`~pdart.pds4.Product` within an :class:`~pdart.pds4.Archive`.
    This is an abstract class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, arch, lid):
        # type: (pdart.pds4.Archive.Archive, LID) -> None
        """
        Create an :class:`~pdart.pds4.Component` given the
        :class:`~pdart.pds4.Archive` it lives in and its
        :class:`~pdart.pds4.LID`.
        """
        assert arch
        self.archive = arch
        assert lid
        assert isinstance(lid, LID)
        self.lid = lid

    def __eq__(self, other):
        return self.archive == other.archive and self.lid == other.lid

    def __str__(self):
        return str(self.lid)

    @abc.abstractmethod
    def absolute_filepath(self):
        # type: () -> unicode
        """
        Return the absolute filepath to the
        :class:`~pdart.pds4.Component`'s directory
        (:class:`~pdart.pds4.Bundle`, :class:`~pdart.pds4.Collection`)
        or file (:class:`~pdart.pds4.Product`).
        """
        pass

    @abc.abstractmethod
    def relative_filepath(self):
        # type: () -> unicode
        """
        Return the relative filepath to the
        :class:`~pdart.pds4.Component`'s directory
        (:class:`~pdart.pds4.Bundle`, :class:`~pdart.pds4.Collection`)
        or file (:class:`~pdart.pds4.Product`).
        """
        pass

    def absolute_filepath_is_directory(self):
        # type: () -> bool
        """
        Return True iff :meth:`absolute_filepath` returns a directory.
        """
        return True

    def files(self):
        # type: () -> Iterator[pdart.pds4.File.File]
        """
        Generate all the files belonging to this
        :class:`~pdart.pds4.Component` as :class:`~pdart.pds4.File`
        objects.
        """
        from pdart.pds4.File import File
        dir = self.relative_filepath()
        for basename in self.archive.root_fs.listdir(dir):
            if basename[0] != '.':
                file = join(dir, basename)
                if self.archive.root_fs.isfile(file):
                    yield File(self, basename)
