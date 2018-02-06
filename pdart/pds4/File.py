"""
Representation of a file belonging to a
:class:`~pdart.pds4.Component`.
"""
from fs.path import basename, join, splitext

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pdart.pds4.Component


# We only import PSD4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.


class File(object):
    """A file belonging to an :class:`~pdart.pds4.Component`."""

    def __init__(self, comp, basename2):
        # type: (pdart.pds4.Component.Component, unicode) -> None
        """
        Create an File given the :class:`~pdart.pds4.Component` it
        belongs to and the basename2 (that is, filepath without
        directory part) for the file.

        Note that this assumes that :class:`~pdart.pds4.Product`s
        don't contain subdirectories.  That won't always be true.
        """
        from pdart.pds4.Product import Product

        assert comp, ('File.__init__() '
                      'where comp = %r and basename2 = %r' %
                      (comp, basename2))
        self.component = comp
        assert basename2, ('File.__init__() '
                           'where comp = %r and basename2 = %r' %
                           (comp, basename2))
        assert basename(basename2) == basename2
        assert splitext(basename2)[1] in Product.FILE_EXTS, basename2
        self.basename = basename2

    def __eq__(self, other):
        return self.component == other.component and \
               self.basename == other.basename

    def __str__(self):
        return '%s in %r' % (self.basename, self.component)

    def __repr__(self):
        return 'File(%r, %r)' % (self.basename, self.component)

    def full_filepath(self):
        # type: () -> unicode
        """Return the full, absolute filepath to the file."""
        if self.component.absolute_filepath_is_directory():
            return join(self.component.absolute_filepath(), self.basename)
        else:
            return self.component.absolute_filepath()
