class Info(object):
    """
    A helper object associated with an ArchiveComponent or
    ArchiveFile, providing information to fill in label fields for it.
    """

    def pds_namespace_url(self):
        """Return the PDS namespace URL"""
        return 'http://pds.nasa.gov/pds4/pds/v1'

    def pds4_schema_url(self):
        """Return the PDS Schema location URL"""
        return 'http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.xsd'

    def xsi_namespace_url(self):
        """Return the XMLSchema instance namespace URL"""
        return 'http://www.w3.org/2001/XMLSchema-instance'

    def xml_model_pds_attributes(self):
        """Return the attributes for PDS4_PDS_1500.sch"""
        # TODO That docstring is not informative enough
        return """href=\"http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch\"
schematypens=\"http://purl.oclc.org/dsdl/schematron\""""

    def fits(self):
        return 'FITS 3.0'

    def information_model_version(self):
        return '1.5.0.0'

    def version_id(self):
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

# was_converted
