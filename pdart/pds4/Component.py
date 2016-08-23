import abc
import os
import os.path

from pdart.pds4.LID import *


class Component(object):
    """
    A :class:`pdart.pds4.Bundle`, :class:`pdart.pds4.Collection`, or
    :class:`pdart.pds4.Product` within an :class:`pdart.pds4.Archive`.
    This is an abstract class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, arch, lid):
        """
        Create an :class:`pdart.pds4.Component` given the
        :class:`pdart.pds4.Archive` it lives in and its
        :class:`pdart.pds4.LID`.
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
        """
        Return the absolute filepath to the
        :class:`pdart.pds4.Component`'s directory
        (:class:`pdart.pds4.Bundle`, :class:`pdart.pds4.Collection`)
        or file (:class:`pdart.pds4.Product`).
        """
        pass

    def absolute_filepath_is_directory(self):
        """
        Return True iff :func:`absolute_filepath()` returns a
        directory.
        """
        return True

    def files(self):
        """
        Generate all the files belonging to this
        :class:`pdart.pds4.Component` as :class:`pdart.pds4.File`
        objects.
        """
        from pdart.pds4.File import File
        dir = self.absolute_filepath()
        for basename in os.listdir(dir):
            if basename[0] != '.':
                file = os.path.join(dir, basename)
                if os.path.isfile(file):
                    yield File(self, basename)
