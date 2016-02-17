import os.path


class ArchiveFile(object):
    def __init__(self, comp, basename):
        assert comp
        self.component = comp
        assert basename
        self.basename = basename

    def __eq__(self, other):
        return self.component == other.component and \
            self.basename == other.basename

    def __str__(self):
        return '%s in %s' % (self.basename, repr(self.component))

    def __repr__(self):
        return 'ArchiveFile(%s, %s)' % (repr(self.basename),
                                        repr(self.component))

    def fullFilepath(self):
        return os.path.join(self.component.directoryFilepath(),
                            self.basename)
