import os.path


class ArchiveFile(object):
    """A file belonging to an ArchiveComponent."""

    def __init__(self, comp, basename):
        """
        Create an ArchiveFile given the component it belongs to and
        the basename (that is, filepath without directory part) for
        the file.

        Note that this assumes that products don't contain
        subdirectories.  That won't always be true.
        """

        assert comp, ('ArchiveFile.__init__() '
                      'where comp = %r and basename = %r' %
                      (comp, basename))
        self.component = comp
        assert basename, ('ArchiveFile.__init__() '
                          'where comp = %r and basename = %r' %
                          (comp, basename))
        assert os.path.basename(basename) == basename
        self.basename = basename

    def __eq__(self, other):
        return self.component == other.component and \
            self.basename == other.basename

    def __str__(self):
        return '%s in %r' % (self.basename, self.component)

    def __repr__(self):
        return 'ArchiveFile(%r, %r)' % (self.basename, self.component)

    def full_filepath(self):
        """Return the full, absolute filepath to the file."""
        if self.component.absolute_filepath_is_directory():
            return os.path.join(self.component.absolute_filepath(),
                                self.basename)
        else:
            return self.component.absolute_filepath()
