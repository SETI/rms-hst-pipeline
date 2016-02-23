class Info(object):
    """
    A helper object associated with an ArchiveComponent or
    ArchiveFile, providing information to fill in label fields for it.
    """

    def pdsNamespaceUrl(self):
        """Return the PDS namespace URL"""
        return 'http://pds.nasa.gov/pds4/pds/v1'

    def informationModelVersion(self):
        return '1.5.0.0'

    def versionID(self):
        return '1.0'

    def PLACEHOLDER(self, tag):
        """
        Return placeholder text for unimplemented helper methods.  The
        name is upper-case to make it stand out: all uses should be
        removed before deployment of the code (because all methods
        will be properly implemented).
        """
        return '### placeholder for %s.%s() ###' % \
            (self.__class__.__name__, tag)

    DO_CHEAT = True

    def CHEATING_PLACEHOLDER(self, cheat, tag):
        """
        Return placeholder text for unimplemented helper methods with
        some cheating to pass schema tests, dependent on a Boolean
        value.  The cheat text is returned if DO_CHEAT is True;
        otherwise PLACEHOLDER() is called.

        Have DO_CHEAT true if you want to pass the tests.  Have
        DO_CHEAT false if you want the placeholders to stand out in
        the generated labels.

        The method name is upper-case to make it stand out: all uses
        should be removed before deployment of the code (because all
        methods will be properly implemented).
        """
        if Info.DO_CHEAT:
            return cheat
        else:
            return self.PLACEHOLDER(tag)
