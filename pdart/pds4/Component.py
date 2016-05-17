import abc
import os
import os.path

from pdart.pds4.LID import *

class Component(object):
    """
    A :class:`Bundle`, :class:`Collection`, or :class:`Product` within
    an :class:`Archive`.  This is an abstract class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, arch, lid):
        """
        Create an :class:`Component` given the :class:`Archive` it
        lives in and its :class:`LID`.
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
        Return the absolute filepath to the :class:`Component`'s
        directory (:class:`Bundle`, :class:`Collection`) or file
        (:class:`Product`).
        """
        pass

    def absolute_filepath_is_directory(self):
        """
        Return True iff :func:`absolute_filepath()` returns a directory.
        """
        return True

    def files(self):
        """
        Generate all the files belonging to this :class:`Component` as
        :class:`File` objects.
        """
        from pdart.pds4.File import File
        dir = self.absolute_filepath()
        for basename in os.listdir(dir):
            if basename[0] != '.':
                file = os.path.join(dir, basename)
                if os.path.isfile(file):
                    yield File(self, basename)
