import os.path


class File(object):
    """A file belonging to an :class:`Component`."""

    def __init__(self, comp, basename):
        """
        Create an File given the :class:`Component` it belongs to and
        the basename (that is, filepath without directory part) for
        the file.

        Note that this assumes that :class:`Product`s don't contain
        subdirectories.  That won't always be true.
        """
        from pdart.pds4.Product import Product

        assert comp, ('File.__init__() '
                      'where comp = %r and basename = %r' %
                      (comp, basename))
        self.component = comp
        assert basename, ('File.__init__() '
                          'where comp = %r and basename = %r' %
                          (comp, basename))
        assert os.path.basename(basename) == basename
        assert os.path.splitext(basename)[1] in Product.FILE_EXTS
        self.basename = basename

    def __eq__(self, other):
        return self.component == other.component and \
            self.basename == other.basename

    def __str__(self):
        return '%s in %r' % (self.basename, self.component)

    def __repr__(self):
        return 'File(%r, %r)' % (self.basename, self.component)

    def full_filepath(self):
        """Return the full, absolute filepath to the file."""
        if self.component.absolute_filepath_is_directory():
            return os.path.join(self.component.absolute_filepath(),
                                self.basename)
        else:
            return self.component.absolute_filepath()
